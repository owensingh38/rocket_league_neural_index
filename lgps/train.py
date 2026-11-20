import contextlib
import importlib
import json
import math
import os
import random
import socket
import time
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from shared import build_optimizers
from .config import LGPSRuntimeConfig, OUTPUT_DIR
from .targets import collapse_probabilities
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from .dataset import LGPSDataset, cleanup_training_cache
from .model import LGPSTransformer
from .evaluate import (
    WandbAdapter, LGPSResults, evaluate_model,
)
from .aux_tasks import (
    LGPSBallStateAux, LGPSGradNorm, LGPSNextTouchAux,
    configured_aux_tasks,
)

class DistributedContext:
    def __init__(self, rank=0, world_size=1, runtime_config=None):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        self.runtime_config = runtime_config or LGPSRuntimeConfig()
        self.is_distributed = self.world_size > 1

        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed rank or world size")

        wants_cpu = self.runtime_config.device == "cpu" or (
            self.runtime_config.device == "auto" and not torch.cuda.is_available()
        )

        if wants_cpu:
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but no CUDA device is available")

            if self.local_rank >= torch.cuda.device_count():
                raise ValueError(f"LOCAL_RANK={self.local_rank} is not a visible GPU")

            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)

            self.device = torch.device("cuda", self.local_rank if self.is_distributed else 0)

        if self.is_distributed and not dist.is_initialized():
            backend = self.runtime_config.ddp_backend if self.device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")

    @property
    def is_main(self):
        return self.rank == 0

    def print(self, *args):
        if self.is_main:
            print(*args)

    def barrier(self):
        if self.is_distributed:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            dist.barrier(device_ids=device_ids)

    def cleanup(self):
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

class BatchDevice:
    @staticmethod
    def move(batch, device):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class BatchPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        stream = torch.cuda.Stream(device=self.device)
        iterator = iter(self.loader)

        def preload(batch):
            if batch is None:
                return None

            with torch.cuda.stream(stream):
                return BatchDevice.move(batch, self.device)

        next_batch = preload(next(iterator, None))

        while next_batch is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(stream)
            batch = next_batch

            for tensor in batch.values():
                if isinstance(tensor, torch.Tensor):
                    tensor.record_stream(current_stream)

            next_batch = preload(next(iterator, None))
            yield batch


class LGPSTrainingMetrics:
    # Accumulate inexpensive metrics during training.

    def __init__(self, device, targets_per_team, horizon_seconds=10.0,
                 two_way_prediction=False):
        self.targets_per_team = targets_per_team
        self.horizon_seconds = horizon_seconds
        self.two_way_prediction = bool(two_way_prediction)
        self.loss = torch.zeros((), device=device, dtype=torch.float64)
        self.primary_loss = torch.zeros((), device=device, dtype=torch.float64)
        self.brier = torch.zeros((), device=device, dtype=torch.float64)
        self.examples = torch.zeros((), device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, loss, logits, targets, primary_loss=None):
        probabilities = collapse_probabilities(
            logits.detach().float().softmax(dim=-1), self.targets_per_team, self.horizon_seconds,
            self.two_way_prediction,
        )
        count = targets.shape[0]
        self.loss += loss.detach().double() * count
        self.primary_loss += (loss if primary_loss is None else primary_loss).detach().double() * count
        target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
        self.brier += (probabilities.square().sum(dim=-1) - 2 * target_probability + 1).sum().double()
        self.examples += count

    def compute(self):
        totals = torch.stack((self.loss, self.primary_loss, self.brier, self.examples))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        loss, primary_loss, brier, examples = totals

        if not examples.item():
            raise RuntimeError("no training examples were observed")

        result = {
            "loss": (loss / examples).item(),
            "primary_loss": (primary_loss / examples).item(),
            "brier": (brier / examples).item(),
        }
        return result


class LGPSTrainer:
    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        test_loader,
        config,
        context,
        model_config,
        runtime_config=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.config = config
        self.model_config = model_config
        self.context = context
        self.device = context.device
        self.runtime_config = runtime_config or LGPSRuntimeConfig(
            device=self.device.type,
            world_size=context.world_size,
        )
        self.batch_size = self.runtime_config.per_gpu_batch_size or config.batch_size
        physical_batch_size = (
            self.batch_size * self.context.world_size
        )
        self.accumulation_steps = max(
            1,
            math.ceil(
                config.effective_batch_size
                / physical_batch_size
            ),
        )
        self.effective_batch_size = (
            physical_batch_size * self.accumulation_steps
        )
        self.results = LGPSResults(OUTPUT_DIR)
        resume_checkpoint = os.environ.get("LGPS_RESUME_CHECKPOINT")
        self.resume_state = (
            torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            if resume_checkpoint else None
        )
        self.aux_task_settings = configured_aux_tasks(config.aux_tasks)
        self.aux_task_names = tuple(
            ([f'ball_{frame}' for frame in self.aux_task_settings[
                LGPSBallStateAux].future_frames]
             if LGPSBallStateAux in self.aux_task_settings else [])
            + (['next_touch'] if LGPSNextTouchAux in self.aux_task_settings else [])
        )
        self.auxiliary_loss_groups = {}

        if LGPSBallStateAux in self.aux_task_settings:
            self.auxiliary_loss_groups['ball_state'] = tuple(
                f'ball_{frame}'
                for frame in self.aux_task_settings[LGPSBallStateAux].future_frames
            )

        if LGPSNextTouchAux in self.aux_task_settings:
            self.auxiliary_loss_groups['next_touch'] = ('next_touch',)

        self.gradnorm = LGPSGradNorm(
            self.aux_task_names, alpha=config.gradnorm_alpha,
            momentum=config.gradnorm_momentum, device=self.device,
        )
        self.training_step = 0
        self.global_examples_seen = 0
        self.wandb_adapter = (
            WandbAdapter(self.runtime_config.wandb_project, self.runtime_config.wandb_entity,
                         self.runtime_config.wandb_run_name, self.runtime_config.wandb_mode,
                         config=WandbAdapter.lgps_config(
                             model_config, config, self.runtime_config,
                             batch_size_per_gpu=self.batch_size,
                             accumulation_steps=self.accumulation_steps,
                             effective_batch_size=self.effective_batch_size,
                         ),
                         run_id=(self.resume_state or {}).get("wandb_run_id"))
            if self.runtime_config.wandb_project is not None and context.is_main else None
        )

        if self.wandb_adapter is not None and self.aux_task_names:
            self.wandb_adapter.configure_auxiliary_plots(
                self.auxiliary_loss_groups, self.auxiliary_loss_groups,
            )

        self.optimizers = build_optimizers(model, config, self.device)
        self.muon_optimizer = self.optimizers[0] if config.optimizer == "Hybrid" else None
        self.adamw_optimizer = self.optimizers[-1] if config.optimizer in ("AdamW", "Hybrid") else None
        # Decay the learning rate smoothly over the requested epoch count.
        self.schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.min_learning_rate,
            )
            for optimizer in self.optimizers
        ]

        # Select the autocast format from the visible CUDA device capability.
        if self.runtime_config.amp_policy == "auto" and self.device.type == "cuda":
            capability = torch.cuda.get_device_capability(self.device)
            self.autocast_dtype = (
                torch.bfloat16
                if capability[0] >= 8
                else torch.float16
            )
        elif self.runtime_config.amp_policy == "auto":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = getattr(torch, self.runtime_config.amp_policy)

        self.amp_enabled = self.device.type == "cuda" and self.autocast_dtype in {torch.float16, torch.bfloat16}

        # FP16 needs gradient scaling; BF16 does not.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.amp_enabled
                and self.autocast_dtype == torch.float16
            ),
        )

        # Validation uses the raw model to avoid uneven DDP forward collectives.

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self, examples):
        correction = self.effective_batch_size / examples

        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)

        if self.scaler.is_enabled():
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
        )

        if self.scaler.is_enabled():
            # GradScaler skips non-finite gradients and adjusts the FP16 scale.
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)

            self.scaler.update()
        elif torch.isfinite(gradient_norm):
            for optimizer in self.optimizers:
                optimizer.step()
        else:
            self.zero_grad()
            raise FloatingPointError("Non-finite gradients without FP16 scaling")

        self.zero_grad()
        return gradient_norm

    def training_batches(self):
        iterator = iter(self.train_loader)
        last_batch = None

        def next_batch():
            batch = next(iterator, None)
            local_count = 0 if batch is None else batch["temporal_targets"].shape[0]

            if local_count == 0:
                return None

            return batch, local_count

        while True:
            group_start = time.perf_counter()
            local_group = [next_batch() for _ in range(self.accumulation_steps)]
            local_available = sum(item is not None for item in local_group)
            shared_available = local_available

            if self.context.is_distributed:
                availability = torch.tensor(
                    local_available, device=self.device, dtype=torch.int64,
                )
                dist.all_reduce(availability, op=dist.ReduceOp.MAX)
                shared_available = int(availability.item())

            if not shared_available:
                return

            if last_batch is None and local_available == 0:
                raise RuntimeError(
                    "A distributed training shard has no batches while another rank has data"
                )
            # The iterable loader stages an accumulation group before yielding
            # its first batch. Attribute that input wait across the group's
            # batches so per-batch throughput includes loader stalls instead
            # of measuring only model execution.
            group_data_wait = time.perf_counter() - group_start

            if self.context.is_distributed:
                wait_tensor = torch.tensor(group_data_wait, device=self.device, dtype=torch.float64)
                dist.all_reduce(wait_tensor, op=dist.ReduceOp.MAX)
                group_data_wait = float(wait_tensor.item())

            group_data_wait /= shared_available
            global_counts = []

            for group_index in range(shared_available):
                current = local_group[group_index] if group_index < len(local_group) else None

                if current is None:
                    _, local_count = last_batch
                    batch = last_batch[0]
                    local_count = 0
                else:
                    batch, local_count = current
                    last_batch = current

                global_count = torch.tensor(local_count, device=self.device, dtype=torch.int64)

                if self.context.is_distributed:
                    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)

                global_counts.append(int(global_count.item()))
                local_group[group_index] = (batch, local_count)

            group_global_count = sum(global_counts)

            for group_index, (batch, local_count) in enumerate(local_group[:shared_available]):
                if self.device.type == "cuda":
                    batch = BatchDevice.move(batch, self.device)

                yield (batch, local_count, global_counts[group_index],
                       group_index + 1 == shared_available, group_data_wait,
                       group_global_count)

            if shared_available < self.accumulation_steps:
                return

    @property
    def raw_model(self):
        # DDP checkpoints should save the underlying model rather than the
        # wrapper so they can be loaded in either distributed or local mode.
        model = (
            self.model.module
            if isinstance(self.model, DDP)
            else self.model
        )

        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        return model

    def forward(self, batch, evaluation=False, model_override=None):
        # Uneven validation shards must not issue DDP forward collectives.
        model = model_override or (self.raw_model if evaluation else self.model)

        def call_model(ball):
            return model(
                ball, batch["players"], batch["goals"], batch["entity_types"],
                batch["entity_teams"], batch["game_context"], batch.get("big_boosts"),
                batch.get("small_boosts"),
                return_aux=not evaluation and bool(self.aux_task_names),
            )

        if (not evaluation and model_override is None and model.training
                and self.config.activation_checkpointing):
            return checkpoint(call_model, batch["ball"], use_reentrant=False)

        return call_model(batch["ball"])

    def auxiliary_losses(self, predictions, batch):
        raw = self.raw_model
        losses = {}

        if LGPSBallStateAux in self.aux_task_settings:
            losses.update(raw.ball_state_aux.losses(predictions, batch['aux_ball_targets']))

        if LGPSNextTouchAux in self.aux_task_settings:
            losses['next_touch'] = raw.next_touch_aux.loss(
                predictions['next_touch'], batch['aux_next_touch_slot'],
                batch['aux_next_touch_seconds'],
            )

        return losses

    def train_epoch(self, epoch):
        self.model.train()
        metrics = LGPSTrainingMetrics(
            self.device, self.model_config.targets_per_team, self.model_config.horizon_seconds,
            self.model_config.two_way_prediction,
        )
        start_time = time.perf_counter()
        last_log_time = start_time
        self.context.print(f"Epoch {epoch}/{self.config.epochs} | training started")
        self.zero_grad()
        gradient_norm = torch.zeros((), device=self.device)
        accumulated_examples = 0
        epoch_examples_seen = 0

        for batch_index, (batch, local_count, global_count, is_last, data_wait,
                          group_global_count) in enumerate(
            self.training_batches(), start=1,
        ):
            batch_start_time = time.perf_counter()

            if batch["ball"].device != self.device:
                batch = BatchDevice.move(batch, self.device)

            if local_count == 0:
                # Keep DDP's forward/backward collectives aligned after this
                # rank exhausts its shard. The repeated input is only a shape
                # template; all sample and auxiliary targets are masked.
                batch = dict(batch)

                for name in ("temporal_targets", "collapsed_targets"):
                    batch[name] = torch.zeros_like(batch[name])

                for name in ("aux_ball_targets", "aux_next_touch_seconds"):
                    if name in batch:
                        batch[name] = torch.full_like(batch[name], float("nan"))

                if "aux_next_touch_slot" in batch:
                    batch["aux_next_touch_slot"] = torch.full_like(
                        batch["aux_next_touch_slot"], -1,
                    )

            accumulated_examples += global_count
            sync_context = (
                self.model.no_sync()
                if self.context.is_distributed and not is_last
                else contextlib.nullcontext()
            )

            with sync_context:
                with torch.autocast(
                    device_type=self.device.type, dtype=self.autocast_dtype,
                    enabled=self.amp_enabled,
                ):
                    output = self.forward(batch)
                    logits, auxiliary_predictions, shared = (
                        output if self.aux_task_names else (output, {}, None)
                    )
                    primary_loss = -(
                        batch["temporal_targets"] * F.log_softmax(logits, dim=-1)
                    ).sum(dim=-1).mean()
                    auxiliary_losses = self.auxiliary_losses(auxiliary_predictions, batch)
                    loss, task_weights = (
                        self.gradnorm.combine(
                            primary_loss, auxiliary_losses, shared,
                        )
                        if self.aux_task_names else (primary_loss, {})
                    )
                    scaled_loss = loss * (
                        local_count * self.context.world_size / max(group_global_count, 1)
                    )

                self.scaler.scale(scaled_loss).backward()

            if is_last:
                gradient_norm = self.optimizer_step(accumulated_examples)
                accumulated_examples = 0

            if local_count:
                metrics.update(loss, logits, batch["collapsed_targets"], primary_loss)

            compute_elapsed = time.perf_counter() - batch_start_time
            batch_elapsed = compute_elapsed + data_wait
            throughput = global_count / batch_elapsed if batch_elapsed > 0 else 0.0
            self.training_step += 1
            self.global_examples_seen += global_count
            epoch_examples_seen += global_count
            auxiliary_task_loss_values = []
            auxiliary_task_weighted_values = []

            for component_names in self.auxiliary_loss_groups.values():
                auxiliary_task_loss_values.append(sum(
                    auxiliary_losses[name][0].detach()
                    for name in component_names
                ))
                auxiliary_task_weighted_values.append(sum(
                    self.gradnorm.weights[self.aux_task_names.index(name)].detach()
                    * auxiliary_losses[name][0].detach()
                    for name in component_names
                ))

            logged_losses = torch.stack(
                [loss.detach(), primary_loss.detach(), *auxiliary_task_loss_values,
                 *auxiliary_task_weighted_values]
            ).float()

            if self.context.is_distributed:
                dist.all_reduce(logged_losses, op=dist.ReduceOp.SUM)
                logged_losses /= self.context.world_size

            auxiliary_task_count = len(self.auxiliary_loss_groups)
            auxiliary_task_weights = {}

            for index, (task_name, component_names) in enumerate(
                self.auxiliary_loss_groups.items(),
            ):
                raw_total = logged_losses[index + 2]
                weighted_total = logged_losses[index + 2 + auxiliary_task_count]

                if raw_total.abs().item() > 1e-12:
                    auxiliary_task_weights[task_name] = (weighted_total / raw_total).item()
                else:
                    auxiliary_task_weights[task_name] = sum(
                        self.gradnorm.weights[self.aux_task_names.index(name)].item()
                        for name in component_names
                    ) / len(component_names)

            if self.context.is_main:
                if self.wandb_adapter is not None:
                    self.wandb_adapter.log_training(
                        {
                            'loss': logged_losses[0].item(),
                            'loss/goal': logged_losses[1].item(),
                            'throughput': throughput,
                            **{f'loss/{task_name}': logged_losses[index + 2].item()
                               for index, task_name in enumerate(self.auxiliary_loss_groups)},
                            **{f'weight/{task_name}': value
                               for task_name, value in auxiliary_task_weights.items()},
                        },
                        self.training_step, epoch,
                    )

            if self.context.is_main and (
                batch_index == 1 or batch_index % self.config.log_interval == 0
            ):
                now = time.perf_counter()
                interval = now - last_log_time
                print(
                    f"Epoch {epoch}/{self.config.epochs} | batch {batch_index:,} | "
                    f"loss {loss.detach().item():.5f} | goal {primary_loss.detach().item():.5f} | "
                    f"grad {float(gradient_norm):.4f} | lr {self.optimizers[-1].param_groups[0]['lr']:.2e} | "
                    f"global batch {global_count:,} | "
                    f"global examples {epoch_examples_seen:,} | "
                    f"throughput {throughput:,.1f} global examples/s | "
                    f"data loading {data_wait:.3f}s/batch | compute {compute_elapsed:.3f}s | "
                    f"{interval:.1f}s since last log"
                )
                last_log_time = now

        if accumulated_examples:
            gradient_norm = self.optimizer_step(accumulated_examples)

        result = metrics.compute()
        elapsed = time.perf_counter() - start_time
        self.context.print(
            f"Epoch {epoch}/{self.config.epochs} | training finished | "
            f"loss {result['loss']:.5f} | goal {result['primary_loss']:.5f} | "
            f"brier {result['brier']:.5f} | "
            f"time {elapsed:.1f}s"
        )
        return result

    def evaluate(self, epoch, loader, label, store_predictions):
        return evaluate_model(self, epoch, loader, label, store_predictions)

    def _rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }

        if self.device.type == "cuda":
            state["torch_cuda"] = torch.cuda.get_rng_state(self.device)

        return state

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])

        if self.device.type == "cuda" and "torch_cuda" in state:
            torch.cuda.set_rng_state(state["torch_cuda"], self.device)

    def _restore_resume_state(self):
        state = self.resume_state

        if state is None:
            return [], float("inf"), 1

        if state["model_config"] != self.raw_model.config.__dict__:
            raise ValueError("Resume checkpoint model config does not match this run")

        if state["train_config"] != self.config.__dict__:
            raise ValueError("Resume checkpoint train config does not match this run")

        if state["runtime_config"] != self.runtime_config.__dict__:
            raise ValueError("Resume checkpoint runtime config does not match this run")

        self.raw_model.load_state_dict(state["model_state_dict"])

        for optimizer, optimizer_state in zip(self.optimizers, state["optimizer_states"]):
            optimizer.load_state_dict(optimizer_state)

        for scheduler, scheduler_state in zip(self.schedulers, state["scheduler_states"]):
            scheduler.load_state_dict(scheduler_state)

        self.gradnorm.initial.copy_(state["gradnorm_initial"].to(self.device))
        self.gradnorm.weights.copy_(state["gradnorm_weights"].to(self.device))
        self.gradnorm.reported_weights = dict(state["gradnorm_reported_weights"])
        self.scaler.load_state_dict(state["scaler_state"])
        self.training_step = int(state["training_step"])
        self.global_examples_seen = int(state["global_examples_seen"])
        self.results.history = list(state["result_history"])
        self.results.epoch_losses = list(state["epoch_losses"])
        self.results.final_metrics = dict(state["final_metrics"])
        self._restore_rng_state(state["rng_states"][self.context.rank])
        self.context.print(f"Resumed training from completed epoch {state['epoch']}")
        history = list(state["history"]) if self.context.is_main else []
        return history, float(state["best_validation_loss"]), int(state["epoch"]) + 1

    def _save_resume_checkpoint(self, epoch, best_validation_loss, history):
        local_rng_state = self._rng_state()

        if self.context.is_distributed:
            rng_states = [None] * self.context.world_size
            torch.distributed.all_gather_object(rng_states, local_rng_state)
        else:
            rng_states = [local_rng_state]

        if not self.context.is_main:
            return

        checkpoint = {
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_states": [optimizer.state_dict() for optimizer in self.optimizers],
            "scheduler_states": [scheduler.state_dict() for scheduler in self.schedulers],
            "gradnorm_initial": self.gradnorm.initial.detach().cpu(),
            "gradnorm_weights": self.gradnorm.weights.detach().cpu(),
            "gradnorm_reported_weights": self.gradnorm.reported_weights,
            "scaler_state": self.scaler.state_dict(),
            "model_config": self.raw_model.config.__dict__,
            "train_config": self.config.__dict__,
            "runtime_config": self.runtime_config.__dict__,
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "history": history,
            "result_history": self.results.history,
            "epoch_losses": self.results.epoch_losses,
            "final_metrics": self.results.final_metrics,
            "training_step": self.training_step,
            "global_examples_seen": self.global_examples_seen,
            "rng_states": rng_states,
            "wandb_run_id": self.wandb_adapter.run.id if self.wandb_adapter else None,
        }
        checkpoint_path = OUTPUT_DIR / "lgps_resume_checkpoint.pt"
        temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(f"Saved resumable checkpoint after epoch {epoch}: {checkpoint_path}")

    def fit(self):
        history, best_validation_loss, start_epoch = self._restore_resume_state()
        requested_stop = os.environ.get("LGPS_STOP_AFTER_EPOCH")
        stop_after_epoch = int(requested_stop) if requested_stop else self.config.epochs

        if stop_after_epoch < start_epoch - 1 or stop_after_epoch > self.config.epochs:
            raise ValueError(
                f"LGPS_STOP_AFTER_EPOCH must be between {start_epoch - 1} and {self.config.epochs}"
            )

        self.context.print(
            "Starting LGPS training"
        )
        self.context.print(
            f"Epochs: {self.config.epochs} | "
            f"Batch size per rank: {self.batch_size} | "
            f"World size: {self.context.world_size} | "
            f"Physical global batch: "
            f"{self.batch_size * self.context.world_size} | "
            f"Gradient accumulation: "
            f"{self.accumulation_steps} | "
            f"Effective global batch: "
            f"{self.effective_batch_size}"
        )

        for epoch in range(start_epoch, stop_after_epoch + 1):
            epoch_start = time.perf_counter()
            epoch_learning_rate = self.optimizers[-1].param_groups[0]['lr']
            epoch_muon_learning_rate = (
                self.muon_optimizer.param_groups[0]['lr']
                if self.muon_optimizer is not None else None
            )

            train_metrics = self.train_epoch(epoch)

            if self.validation_loader is None:
                validation_metrics = {
                    key: None
                    for key in train_metrics
                }
                probabilities = np.empty((0, 3), dtype=np.float32)
                targets = np.empty((0,), dtype=np.int64)
            else:
                validation_metrics, (probabilities, targets) = self.evaluate(
                    epoch,
                    self.validation_loader,
                    "validation",
                    store_predictions=True,
                )

            for scheduler in self.schedulers:
                scheduler.step()

            if self.context.is_main:
                row = self.results.update(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    evaluation_metrics=validation_metrics,
                    probabilities=probabilities,
                    targets=targets,
                )
                row["learning_rate"] = epoch_learning_rate

                if self.muon_optimizer is not None:
                    row["learning_rate"] = epoch_learning_rate
                    row["muon_learning_rate"] = epoch_muon_learning_rate

                history.append(row)

                if self.wandb_adapter is not None:
                    if self.validation_loader is not None:
                        self.wandb_adapter.log_validation(
                            row, epoch, probabilities=probabilities, targets=targets
                        )
                        self.wandb_adapter.log_checkpoint_plots(OUTPUT_DIR, epoch)

            if self.context.is_main:
                print(
                    f"Epoch {epoch}/{self.config.epochs} complete in "
                    f"{time.perf_counter() - epoch_start:.1f}s"
                )
                print(
                    "Full epoch metrics:\n"
                    + json.dumps(
                        LGPSResults._json_value({
                            k: v for k, v in LGPSResults._reported_metrics(row).items()
                            if not k.endswith("calibration_bins")
                        }),
                        indent=2,
                    )
                )

            # Only rank 0 writes the checkpoint to avoid simultaneous writes.
            if (
                self.context.is_main
                and self.validation_loader is not None
                and validation_metrics["loss"] < best_validation_loss
            ):
                best_validation_loss = validation_metrics["loss"]

                checkpoint_path = (
                    OUTPUT_DIR / "lgps_transformer.pt"
                )

                torch.save(
                    {
                        "model_state_dict": self.raw_model.state_dict(),
                        "gradnorm_initial": self.gradnorm.initial.cpu(),
                        "gradnorm_weights": self.gradnorm.weights.cpu(),
                        "model_config": self.raw_model.config.__dict__,
                        "train_config": self.config.__dict__,
                        "epoch": epoch,
                        "validation_loss": best_validation_loss,
                    },
                    checkpoint_path,
                )
                print(
                    f"Saved new best checkpoint: {checkpoint_path} | "
                    f"validation loss {best_validation_loss:.5f}"
                )

            if self.config.store_model_checkpoint and self.validation_loader is not None:
                self._save_resume_checkpoint(epoch, best_validation_loss, history)

            # Keep ranks aligned at epoch boundaries.
            self.context.barrier()

        run_complete = stop_after_epoch >= self.config.epochs

        if self.test_loader is not None and run_complete:
            test_metrics, (test_probabilities, test_targets) = self.evaluate(
                self.config.epochs,
                self.test_loader,
                "test",
                store_predictions=True,
            )

            if self.context.is_main:
                test_row = self.results.update(
                    epoch=self.config.epochs,
                    train_metrics=None,
                    evaluation_metrics=test_metrics,
                    probabilities=test_probabilities,
                    targets=test_targets,
                    label="test",
                )
                print(
                    "Full test metrics:\n"
                    + json.dumps(
                        LGPSResults._json_value({
                            k: v for k, v in LGPSResults._reported_metrics(
                                self.results.final_metrics
                            ).items()
                            if not k.endswith("calibration_bins")
                        }),
                        indent=2,
                    )
                )
                metrics_path = OUTPUT_DIR / "metrics.json"
                metrics_path.write_text(
                    json.dumps(
                        LGPSResults._json_value({
                            **LGPSResults._metrics_payload(
                                self.results.final_metrics, self.results.history
                            ),
                        }),
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                if self.wandb_adapter is not None:
                    self.wandb_adapter.log_test(test_row, OUTPUT_DIR, metrics_path)

        if self.context.is_main and self.validation_loader is None and run_complete:
            checkpoint_path = OUTPUT_DIR / "lgps_transformer.pt"
            torch.save(
                {
                    "model_state_dict": self.raw_model.state_dict(),
                    "gradnorm_initial": self.gradnorm.initial.cpu(),
                    "gradnorm_weights": self.gradnorm.weights.cpu(),
                    "model_config": self.raw_model.config.__dict__,
                    "train_config": self.config.__dict__,
                    "epoch": self.config.epochs,
                    "validation_loss": None,
                },
                checkpoint_path,
            )
            print(f"Saved final checkpoint: {checkpoint_path}")

        if self.wandb_adapter is not None:
            self.wandb_adapter.finish()

        self.context.print(
            "Training complete" if run_complete else
            f"Training paused at completed epoch {stop_after_epoch}"
        )

        if self.context.is_main:
            return pl.DataFrame(history)

        return None

def create_model(model_config, train_config):
    return LGPSTransformer(
        config=model_config,
        aux_tasks=train_config.aux_tasks,
    )


def prepare_model(model_config, train_config, runtime_config, context):
    model = create_model(model_config, train_config).to(context.device)

    if runtime_config.compile_mode is not None:
        model = torch.compile(model, mode=runtime_config.compile_mode, dynamic=False)

    if context.is_distributed:
        ddp_kwargs = {
            "device_ids": [context.local_rank] if context.device.type == "cuda" else None,
            "output_device": context.local_rank if context.device.type == "cuda" else None,
            "broadcast_buffers": False,
            # Excluded entity encoders are not constructed, so every parameter
            # participates in every step and DDP can skip its costly graph walk.
            "find_unused_parameters": runtime_config.find_unused_parameters,
            "gradient_as_bucket_view": True,
            # static_graph is incompatible with this no_sync gradient
            # accumulation path.
            "static_graph": False,
        }
        model = DDP(model, **ddp_kwargs)

    return model


def build_trainer(model_config, train_config, context, runtime_config=None):
    runtime_config = runtime_config or context.runtime_config
    process_seed = (train_config.seed + context.rank) % (2 ** 32)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    loaders = LGPSDataset(
        train_config,
        model_config,
        runtime_config,
        rank=context.rank,
        world_size=context.world_size,
    )
    return LGPSTrainer(
        model=prepare_model(model_config, train_config, runtime_config, context),
        train_loader=loaders.train_loader(),
        validation_loader=loaders.validation_loader(),
        test_loader=loaders.test_loader(),
        config=train_config,
        context=context,
        model_config=model_config,
        runtime_config=runtime_config,
    )


def ddp_worker(rank, world_size, model_config, train_config, runtime_config, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    context = DistributedContext(rank, world_size, runtime_config)
    current_device = (
        torch.cuda.current_device() if context.device.type == "cuda" else "cpu"
    )
    print(
        f"Runtime: rank={context.rank} local_rank={context.local_rank} "
        f"device={context.device} current_device={current_device} "
        f"distributed={context.is_distributed} world_size={context.world_size}",
    )

    try:
        trainer = build_trainer(model_config, train_config, context, runtime_config)
        return trainer.fit()
    finally:
        context.cleanup()


def _ddp_bootstrap(
    rank,
    world_size,
    worker,
    model_config,
    train_config,
    runtime_config,
    master_port,
):
    worker(
        rank,
        world_size,
        model_config,
        train_config,
        runtime_config,
        master_port,
    )


def train_lgps(model_config, train_config, runtime_config):
    if runtime_config.world_size < 1:
        raise ValueError("world_size must be positive")

    if runtime_config.device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")

    available_gpus = torch.cuda.device_count()
    uses_cuda = runtime_config.device == "cuda" or (
        runtime_config.device == "auto" and torch.cuda.is_available()
    )

    if uses_cuda and runtime_config.world_size > available_gpus:
        raise RuntimeError(
            f"runtime requests {runtime_config.world_size} GPU(s), "
            f"but only {available_gpus} are visible"
        )

    world_size = (
        runtime_config.world_size
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if world_size == 1:
        context = DistributedContext(0, 1, runtime_config)

        try:
            trainer = build_trainer(model_config, train_config, context, runtime_config)
            history = trainer.fit()
            return history, trainer.results.final_metrics
        finally:
            try:
                context.cleanup()
            finally:
                cleanup_training_cache(train_config)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = sock.getsockname()[1]

    # Load PyTorch's worker reducers before spawning distributed processes.
    importlib.import_module("torch._utils")

    try:
        mp.spawn(
            _ddp_bootstrap,
            args=(
                world_size,
                ddp_worker,
                model_config,
                train_config,
                runtime_config,
                master_port,
            ),
            nprocs=world_size,
            join=True,
        )
    finally:
        cleanup_training_cache(train_config)

    metrics_path = OUTPUT_DIR / "metrics.json"
    
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return pl.DataFrame(payload["history"]), payload["latest"]
    
    return None, None
