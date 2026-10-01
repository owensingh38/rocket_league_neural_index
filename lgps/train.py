import contextlib
import importlib
import inspect
import json
import math
import os
import socket
import subprocess
import sys
import time
import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from props import LGPSRuntimeConfig, OUTPUT_DIR, collapse_probabilities
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from .dataset import LGPSData
from .model import LGPSTransformer


plt.switch_backend("Agg")

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

    def print(self, *args, **kwargs):
        if self.is_main:
            kwargs.setdefault("flush", True)
            print(*args, **kwargs)

    def barrier(self):
        if self.is_distributed:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            dist.barrier(device_ids=device_ids)

    def cleanup(self):
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

class WandbAdapter:
    # Optional Weights & Biases reporting adapter.
    def __init__(self, project, entity=None, name=None, mode=None):
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as error:
            raise ImportError("W&B reporting requires `pip install wandb`") from error
        self.wandb = wandb
        self.run = wandb.init(project=project, entity=entity, name=name, mode=mode)

    def log_evaluation(self, metrics, epoch, split):
        payload = {"epoch": int(epoch)}
        for key, value in metrics.items():
            if key != "calibration_bins" and isinstance(value, (int, float, np.number)):
                payload[f"{split}/{key}"] = value
        if metrics.get("calibration_bins") is not None:
            table = self.wandb.Table(columns=["horizon_seconds", "channel", "bin", "count", "mean_prediction", "mean_outcome"])
            for horizon, horizon_bins in enumerate(metrics["calibration_bins"], start=1):
                for channel, channel_bins in enumerate(horizon_bins):
                    for bin_index, (count, prediction_sum, outcome_sum) in enumerate(channel_bins):
                        count = float(count)
                        table.add_data(horizon, ("confidence", "blue", "orange", "neither")[channel], bin_index,
                                       count, float(prediction_sum) / count if count else None,
                                       float(outcome_sum) / count if count else None)
            payload[f"{split}/calibration_bins"] = table
        self.run.log(payload, step=int(epoch))


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
            torch.cuda.current_stream(self.device).wait_stream(stream)
            batch = next_batch
            next_batch = preload(next(iterator, None))
            yield batch


class LGPSClassificationMetrics:
    def __init__(self, device, targets_per_team, store_predictions=False):
        self.device = device
        self.targets_per_team = targets_per_team
        self.store_predictions = store_predictions

        # Keep metric sums rather than storing every prediction for the epoch.
        self.loss_sum = torch.zeros(
            (),
            device=device,
            dtype=torch.float64,
        )
        self.correct = torch.zeros(
            (),
            device=device,
            dtype=torch.float64,
        )
        self.collapsed_three_way_loss_sum = torch.zeros(
            (),
            device=device,
            dtype=torch.float64,
        )
        self.brier_sum = torch.zeros(
            (),
            device=device,
            dtype=torch.float64,
        )
        self.count = torch.zeros(
            (),
            device=device,
            dtype=torch.float64,
        )
        self.probabilities = []
        self.targets = []

    @torch.no_grad()
    def update(self, loss, logits, targets, _unused=None):
        probabilities = collapse_probabilities(logits.detach().float().softmax(dim=-1), self.targets_per_team)
        n = targets.shape[0]

        # Report the same joint soft temporal CE optimized by training.
        self.loss_sum += loss.detach().double() * n

        self.correct += (
            probabilities.argmax(dim=-1) == targets
        ).sum().double()

        self.collapsed_three_way_loss_sum += F.nll_loss(
            probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
            targets,
            reduction="sum",
        ).double()

        # For a one-hot target, ||p - y||^2 = ||p||^2 - 2 p_y + 1.
        target_probability = probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)
        self.brier_sum += (
            probabilities.square().sum(dim=-1)
            - 2.0 * target_probability
            + 1.0
        ).sum().double()

        self.count += n

        if self.store_predictions:
            # Keep validation outputs on-device until the final distributed gather.
            self.probabilities.append(probabilities)
            self.targets.append(targets.detach())

    def arrays(self):
        if not self.store_predictions or not self.probabilities:
            local_probabilities = torch.empty(
                (0, 3),
                device=self.device,
                dtype=torch.float32,
            )
            local_targets = torch.empty(
                (0,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            local_probabilities = torch.cat(
                self.probabilities
            )
            local_targets = torch.cat(
                self.targets
            )

        if dist.is_available() and dist.is_initialized():
            local_size = torch.tensor(
                [local_targets.shape[0]],
                device=self.device,
                dtype=torch.long,
            )
            sizes = [
                torch.zeros_like(local_size)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(sizes, local_size)
            max_size = max(int(size.item()) for size in sizes)

            padded_probabilities = torch.zeros(
                (max_size, 3),
                device=self.device,
                dtype=torch.float32,
            )
            padded_targets = torch.zeros(
                (max_size,),
                device=self.device,
                dtype=torch.long,
            )
            padded_probabilities[:local_size.item()] = local_probabilities
            padded_targets[:local_size.item()] = local_targets

            gathered_probabilities = [
                torch.empty_like(padded_probabilities)
                for _ in range(dist.get_world_size())
            ]
            gathered_targets = [
                torch.empty_like(padded_targets)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_probabilities, padded_probabilities)
            dist.all_gather(gathered_targets, padded_targets)

            probabilities = torch.cat(
                [
                    values[:int(size.item())]
                    for values, size in zip(gathered_probabilities, sizes)
                ]
            )
            targets = torch.cat(
                [
                    values[:int(size.item())]
                    for values, size in zip(gathered_targets, sizes)
                ]
            )
        else:
            probabilities = local_probabilities
            targets = local_targets

        return probabilities.cpu().numpy(), targets.cpu().numpy()

    def compute(self):
        # Reduce metric totals across all DDP ranks before reporting them.
        totals = torch.stack(
            (
                self.loss_sum,
                self.correct,
                self.collapsed_three_way_loss_sum,
                self.brier_sum,
                self.count,
            )
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                totals,
                op=dist.ReduceOp.SUM,
            )

        loss_sum, correct, collapsed_three_way_loss_sum, brier_sum, count = totals

        if count.item() == 0:
            raise RuntimeError("The data split produced zero usable rows.")

        return {
            "loss": (loss_sum / count).item(),
            "collapsed_three_way_loss": (collapsed_three_way_loss_sum / count).item(),
            "accuracy": (correct / count).item(),
            "brier": (brier_sum / count).item(),
            "examples": int(count.item()),
        }


class LGPSTrainingMetrics:
    # GPU-resident inexpensive training metrics.
    #
    #     Exact sklearn metrics remain validation/test
    #     work and are deliberately absent from the per-batch training hot path.
    #     
    def __init__(self, device, targets_per_team):
        self.targets_per_team = targets_per_team
        self.loss = torch.zeros((), device=device, dtype=torch.float64)
        self.brier = torch.zeros((), device=device, dtype=torch.float64)
        self.examples = torch.zeros((), device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, loss, logits, targets):
        probabilities = collapse_probabilities(logits.detach().float().softmax(dim=-1), self.targets_per_team)
        count = targets.shape[0]
        self.loss += loss.detach().double() * count
        target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
        self.brier += (probabilities.square().sum(dim=-1) - 2 * target_probability + 1).sum().double()
        self.examples += count

    def compute(self):
        if not self.examples.item():
            raise RuntimeError("no training examples were observed")
        return {
            "loss": (self.loss / self.examples).item(),
            "brier": (self.brier / self.examples).item(),
        }


class LGPSResults:
    CLASS_NAMES = ("blue", "orange", "neither")

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_losses = []
        self.history = []
        self.final_metrics = {}

    @staticmethod
    def _json_value(value):
        if isinstance(value, dict):
            return {
                str(key): LGPSResults._json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [LGPSResults._json_value(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _safe_auc(metric, labels, scores):
        if np.unique(labels).size < 2:
            return None
        return float(metric(labels, scores))

    @staticmethod
    def _calibration_curve(labels, scores, bins=10):
        edges = np.linspace(0.0, 1.0, bins + 1)
        indices = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
        mean_scores = []
        fractions = []

        for index in range(bins):
            selected = indices == index
            if selected.any():
                mean_scores.append(float(scores[selected].mean()))
                fractions.append(float(labels[selected].mean()))

        return np.asarray(mean_scores), np.asarray(fractions)

    def _write_plots(self, probabilities, targets):
        if probabilities.size == 0:
            return

        names = self.CLASS_NAMES
        colors = ("#188cff", "#ff8c18", "#777777")

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(
            range(1, len(self.epoch_losses) + 1),
            self.epoch_losses,
            color="#333333",
            linewidth=1.8,
            marker="o",
        )
        axis.set(
            title="LGPS Training Loss by Epoch",
            xlabel="Epoch",
            ylabel="Cross-entropy loss",
        )
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(self.output_dir / "epoch_losses.png", dpi=120)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 6))
        for index, (name, color) in enumerate(zip(names, colors)):
            labels = (targets == index).astype(np.int32)
            if np.unique(labels).size < 2:
                continue
            axis.plot(
                *self._roc_curve(labels, probabilities[:, index]),
                color=color,
                linewidth=1.8,
                label=f"{name} AUC={self._safe_auc(roc_auc_score, labels, probabilities[:, index]):.3f}",
            )
        axis.plot((0.0, 1.0), (0.0, 1.0), "--", color="#999999")
        axis.set(title="LGPS ROC-AUC", xlabel="False Positive Rate", ylabel="True Positive Rate")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(self.output_dir / "lgps_roc_auc.png", dpi=120)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 6))
        for index, (name, color) in enumerate(zip(names, colors)):
            labels = (targets == index).astype(np.int32)
            if np.unique(labels).size < 2:
                continue
            precision, recall = self._pr_curve(labels, probabilities[:, index])
            score = self._safe_auc(average_precision_score, labels, probabilities[:, index])
            axis.plot(recall, precision, color=color, linewidth=1.8, label=f"{name} AP={score:.3f}")
        axis.set(title="LGPS Precision-Recall AUC", xlabel="Recall", ylabel="Precision")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(self.output_dir / "lgps_pr_auc.png", dpi=120)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 8))
        axis.plot((0.0, 1.0), (0.0, 1.0), "--", color="#999999", label="Perfect calibration")
        for index, (name, color) in enumerate(zip(names, colors)):
            labels = (targets == index).astype(np.float32)
            mean_scores, fractions = self._calibration_curve(labels, probabilities[:, index])
            axis.plot(mean_scores, fractions, "o-", color=color, label=name)
        axis.set(title="LGPS Calibration", xlabel="Mean Predicted Probability", ylabel="Observed Frequency")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(self.output_dir / "lgps_calibration.png", dpi=120)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 6))
        for index, (name, color) in enumerate(zip(names, colors)):
            axis.hist(
                probabilities[:, index],
                bins=25,
                range=(0.0, 1.0),
                alpha=0.45,
                color=color,
                label=name,
            )
        axis.set(title="LGPS Probability Distribution", xlabel="Predicted Probability", ylabel="Frames")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(self.output_dir / "lgps_probability_distribution.png", dpi=120)
        plt.close(figure)

    @staticmethod
    def _roc_curve(labels, scores):
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        return false_positive_rate, true_positive_rate

    @staticmethod
    def _pr_curve(labels, scores):
        precision, recall, _ = precision_recall_curve(labels, scores)
        return precision, recall

    def update(self, epoch, train_metrics, evaluation_metrics, probabilities, targets, label="validation"):
        if train_metrics is not None:
            self.epoch_losses.append(
                float(train_metrics["loss"])
            )

        metrics = {
            "epoch": int(epoch),
            **({f"train_{key}": value for key, value in train_metrics.items()} if train_metrics is not None else {}),
            **{f"{label}_{key}": value for key, value in evaluation_metrics.items()},
        }

        for index, name in enumerate(self.CLASS_NAMES):
            labels = (targets == index).astype(np.int32)
            scores = probabilities[:, index]
            metrics[f"{label}_{name}_roc_auc"] = self._safe_auc(roc_auc_score, labels, scores)
            metrics[f"{label}_{name}_pr_auc"] = self._safe_auc(average_precision_score, labels, scores)

        metrics[f"{label}_macro_roc_auc"] = float(
            np.nanmean([
                metrics[f"{label}_{name}_roc_auc"]
                for name in self.CLASS_NAMES
                if metrics[f"{label}_{name}_roc_auc"] is not None
            ])
        ) if any(metrics[f"{label}_{name}_roc_auc"] is not None for name in self.CLASS_NAMES) else None
        metrics[f"{label}_macro_pr_auc"] = float(
            np.nanmean([
                metrics[f"{label}_{name}_pr_auc"]
                for name in self.CLASS_NAMES
                if metrics[f"{label}_{name}_pr_auc"] is not None
            ])
        ) if any(metrics[f"{label}_{name}_pr_auc"] is not None for name in self.CLASS_NAMES) else None

        self.history.append(metrics)
        self.final_metrics = metrics
        self._write_plots(probabilities, targets)

        payload = {
            "latest": metrics,
            "history": self.history,
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(self._json_value(payload), indent=2),
            encoding="utf-8",
        )

        return metrics

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
        self.validation_batch_size = config.validation_batch_size
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
        self.wandb_adapter = (
            WandbAdapter(runtime_config.wandb_project, runtime_config.wandb_entity,
                         runtime_config.wandb_run_name, runtime_config.wandb_mode)
            if runtime_config.wandb_project is not None and context.is_main else None
        )

        # AdamW is the main optimizer for the transformer parameters.
        optimizer_kwargs = {
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
        if self.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        try:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                **optimizer_kwargs,
            )
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                **optimizer_kwargs,
            )

        # Decay the learning rate smoothly over the requested epoch count.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.min_learning_rate,
        )

        # T4 GPUs use FP16 Tensor Cores; newer GPUs can use BF16.
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
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self, examples):
        # DDP averages rank gradients; normalize by the actual accumulated rows.
        correction = self.effective_batch_size / examples
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)

        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
        )
        if self.scaler.is_enabled():
            # GradScaler skips non-finite gradients and adjusts the FP16 scale.
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif torch.isfinite(gradient_norm):
            self.optimizer.step()
        else:
            self.zero_grad()
            raise FloatingPointError("Non-finite gradients without FP16 scaling")

        self.zero_grad()
        return gradient_norm

    def training_batches(self):
        iterator = iter(BatchPrefetcher(self.train_loader, self.device))

        def next_batch():
            batch = next(iterator, None)
            local_count = 0 if batch is None else batch["temporal_targets"].shape[0]
            if local_count == 0:
                return None
            return batch, local_count, local_count * self.context.world_size

        while True:
            local_group = [next_batch() for _ in range(self.accumulation_steps)]
            local_available = sum(item is not None for item in local_group)
            shared_available = local_available
            if self.context.is_distributed:
                availability = torch.tensor(
                    local_available, device=self.device, dtype=torch.int64,
                )
                dist.all_reduce(availability, op=dist.ReduceOp.MIN)
                shared_available = int(availability.item())
            if not shared_available:
                return
            # Iterable shards can differ by a few final rows. Limit the final
            # accumulation group to the shortest rank so every DDP backward
            # issues the same collectives; discard at most three batches/rank.
            for group_index, current in enumerate(local_group[:shared_available]):
                yield (*current, group_index + 1 == shared_available)
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
        return model

    def forward(self, batch, evaluation=False):
        # Uneven validation shards must not issue DDP forward collectives.
        model = self.raw_model if evaluation else self.model
        def call_model(ball):
            return model(
                ball, batch["players"], batch["goals"], batch["entity_types"],
                batch["entity_teams"], batch["game_context"], batch.get("big_boosts"),
                batch.get("small_boosts"),
            )
        if not evaluation and model.training and self.runtime_config.activation_checkpointing:
            return checkpoint(call_model, batch["ball"], use_reentrant=False)
        return call_model(batch["ball"])

    def train_epoch(self, epoch):
        self.model.train()

        metrics = LGPSTrainingMetrics(self.device, self.model_config.targets_per_team)
        start_time = time.perf_counter()
        last_log_time = start_time
        local_examples = 0

        self.context.print(
            f"Epoch {epoch}/{self.config.epochs} | training started"
        )

        self.zero_grad()
        gradient_norm = torch.zeros(
            (),
            device=self.device,
        )

        accumulated_examples = 0
        for batch_index, (batch, local_count, global_count, is_last) in enumerate(
            self.training_batches(), start=1,
        ):
            if batch["ball"].device != self.device:
                batch = BatchDevice.move(
                    batch,
                    self.device,
                )

            accumulated_examples += global_count
            synchronize = is_last
            sync_context = (
                self.model.no_sync()
                if self.context.is_distributed and not synchronize
                else contextlib.nullcontext()
            )

            # DDP synchronizes only the final microbatch in each accumulation
            # group; the local gradients are accumulated without NCCL traffic.
            with sync_context:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.amp_enabled,
                ):
                    logits = self.forward(batch)

                    raw_loss = -(batch["temporal_targets"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                    loss = raw_loss * (
                        local_count * self.context.world_size / self.effective_batch_size
                    )

                self.scaler.scale(loss).backward()

            if synchronize:
                # Update once per effective batch, including a final partial
                # group whose last backward pass performed synchronization.
                gradient_norm = self.optimizer_step(accumulated_examples)
                accumulated_examples = 0

            if local_count:
                metrics.update(raw_loss, logits, batch["collapsed_targets"])
            local_examples += local_count

            # Only rank 0 prints progress to avoid duplicate console output.
            if (
                self.context.is_main
                and batch_index % self.config.log_interval == 0
            ):
                now = time.perf_counter()
                elapsed = now - start_time
                interval = now - last_log_time
                global_examples_per_second = (
                    local_examples * self.context.world_size / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"batch {batch_index:,} | "
                    f"loss {raw_loss.detach().item():.5f} | "
                    f"grad {float(gradient_norm):.4f} | "
                    f"lr {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"local examples {local_examples:,} | "
                    f"{global_examples_per_second:,.1f} global examples/s | "
                    f"{interval:.1f}s since last log"
                )

                last_log_time = now

        if accumulated_examples:
            # Every rank applies the same final partial accumulation group.
            gradient_norm = self.optimizer_step(accumulated_examples)

        result = metrics.compute()
        elapsed = time.perf_counter() - start_time

        self.context.print(
            f"Epoch {epoch}/{self.config.epochs} | training finished | "
            f"loss {result['loss']:.5f} | "
            f"brier {result['brier']:.5f} | "
            f"time {elapsed:.1f}s"
        )

        return result

    @torch.inference_mode()
    def evaluate(self, epoch, loader, label, store_predictions):
        self.model.eval()

        metrics = LGPSClassificationMetrics(
            self.device,
            self.model_config.targets_per_team,
            store_predictions=store_predictions,
        )

        start_time = time.perf_counter()

        self.context.print(
            f"Epoch {epoch}/{self.config.epochs} | {label} started"
        )

        for batch_index, batch in enumerate(loader, start=1):
            batch = BatchDevice.move(batch, self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.forward(batch, evaluation=True)
                loss = -(batch["temporal_targets"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

            metrics.update(loss, logits, batch["collapsed_targets"])

            if (
                self.context.is_main
                and batch_index % self.config.log_interval == 0
            ):
                print(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"{label} batch {batch_index:,}"
                )

        result = metrics.compute()
        elapsed = time.perf_counter() - start_time
        global_examples_per_second = (
            result["examples"] / elapsed
            if elapsed > 0
            else 0.0
        )
        per_device_examples_per_second = (
            global_examples_per_second / self.context.world_size
        )

        self.context.print(
            f"Epoch {epoch}/{self.config.epochs} | {label} finished | "
            f"loss {result['loss']:.5f} | "
            f"accuracy {result['accuracy']:.4f} | "
            f"collapsed three-way loss {result['collapsed_three_way_loss']:.5f} | "
            f"brier {result['brier']:.5f} | "
            f"examples {result['examples']:,} | "
            f"time {elapsed:.1f}s | "
            f"throughput {per_device_examples_per_second:,.1f}/device "
            f"({global_examples_per_second:,.1f} global/s)"
        )

        return result, metrics.arrays()

    def fit(self):
        history = []
        best_validation_loss = float("inf")

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

        for epoch in range(
            1,
            self.config.epochs + 1,
        ):
            epoch_start = time.perf_counter()

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

            self.scheduler.step()

            if self.context.is_main:
                row = self.results.update(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    evaluation_metrics=validation_metrics,
                    probabilities=probabilities,
                    targets=targets,
                )
                row["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                history.append(row)
                if self.wandb_adapter is not None:
                    self.wandb_adapter.log_evaluation(train_metrics, epoch, "train")
                    if self.validation_loader is not None:
                        self.wandb_adapter.log_evaluation(validation_metrics, epoch, "validation")

            if self.context.is_main:
                print(
                    f"Epoch {epoch}/{self.config.epochs} complete in "
                    f"{time.perf_counter() - epoch_start:.1f}s"
                )
                print(
                    "Full epoch metrics:\n"
                    + json.dumps(
                        LGPSResults._json_value({k:v for k,v in row.items() if not k.endswith("calibration_bins")}),
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

            # Keep ranks aligned at epoch boundaries.
            self.context.barrier()

        if self.test_loader is not None:
            test_metrics, (test_probabilities, test_targets) = self.evaluate(
                self.config.epochs,
                self.test_loader,
                "test",
                store_predictions=True,
            )
            if self.context.is_main:
                self.results.update(
                    epoch=self.config.epochs,
                    train_metrics=None,
                    evaluation_metrics=test_metrics,
                    probabilities=test_probabilities,
                    targets=test_targets,
                    label="test",
                )
                if self.wandb_adapter is not None:
                    self.wandb_adapter.log_evaluation(test_metrics, self.config.epochs, "test")
                print(
                    "Full test metrics:\n"
                    + json.dumps(
                        LGPSResults._json_value({k:v for k,v in self.results.final_metrics.items() if not k.endswith("calibration_bins")}),
                        indent=2,
                    )
                )
                metrics_path = OUTPUT_DIR / "metrics.json"
                metrics_path.write_text(
                    json.dumps(
                        LGPSResults._json_value({
                            "latest": self.results.final_metrics,
                            "history": self.results.history,
                        }),
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        if self.context.is_main and self.validation_loader is None:
            checkpoint_path = OUTPUT_DIR / "lgps_transformer.pt"
            torch.save(
                {
                    "model_state_dict": self.raw_model.state_dict(),
                    "model_config": self.raw_model.config.__dict__,
                    "train_config": self.config.__dict__,
                    "epoch": self.config.epochs,
                    "validation_loss": None,
                },
                checkpoint_path,
            )
            print(f"Saved final checkpoint: {checkpoint_path}")

        self.context.print(
            "Training complete"
        )

        if self.context.is_main:
            return pl.DataFrame(history)

        return None

def create_model(model_config, train_config):
    return LGPSTransformer(config=model_config)


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
            # static_graph is incompatible with no_sync gradient accumulation
            # on the PyTorch build used by Kaggle/Modal T4 workers.
            "static_graph": False,
        }
        # Newer PyTorch releases batch gradient copies into each ready bucket.
        # Keep the notebook usable on Kaggle images with older DDP signatures.
        if "batched_grad_copy" in inspect.signature(DDP).parameters:
            ddp_kwargs["batched_grad_copy"] = True
        model = DDP(model, **ddp_kwargs)
    return model


def build_trainer(model_config, train_config, context, runtime_config=None):
    runtime_config = runtime_config or context.runtime_config
    data = LGPSData(
        train_config,
        model_config,
        runtime_config,
        rank=context.rank,
        world_size=context.world_size,
    )
    return LGPSTrainer(
        model=prepare_model(model_config, train_config, runtime_config, context),
        train_loader=data.train_loader(),
        validation_loader=data.validation_loader(),
        test_loader=data.test_loader(),
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
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    context = DistributedContext(rank, world_size, runtime_config)
    context.print(
        f"Runtime: device={context.device} | distributed={context.is_distributed} | "
        f"world_size={context.world_size}"
    )
    try:
        trainer = build_trainer(model_config, train_config, context, runtime_config)
        return trainer.fit()
    finally:
        context.cleanup()


_DDP_BOOTSTRAP = r"""
import cloudpickle
import os

worker, model_config, train_config, runtime_config, master_port = cloudpickle.load(
    __import__("sys").stdin.buffer
)
worker(
    int(os.environ["RANK"]),
    int(os.environ["WORLD_SIZE"]),
    model_config,
    train_config,
    runtime_config,
    int(master_port),
)
"""


def train_lgps(model_config, train_config, runtime_config):
    available_gpus = torch.cuda.device_count()
    if (runtime_config.device == "cuda" or runtime_config.world_size > 1) and runtime_config.world_size > available_gpus:
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
            context.cleanup()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = sock.getsockname()[1]

    # Notebook-defined worker functions capture classes with constant tensors.
    # Some Kaggle PyTorch builds lazily omit this reducer dependency, causing
    # cloudpickle to fail with ``torch has no attribute _utils`` before DDP
    # starts. Import it explicitly before serializing the worker payload.
    importlib.import_module("torch._utils")
    payload = cloudpickle.dumps(
        (ddp_worker, model_config, train_config, runtime_config, master_port)
    )
    workers = []
    try:
        for rank in range(world_size):
            environment = os.environ.copy()
            environment.update(
                {
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(master_port),
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            worker = subprocess.Popen(
                [sys.executable, "-u", "-c", _DDP_BOOTSTRAP],
                env=environment,
                stdin=subprocess.PIPE,
            )
            workers.append(worker)
            worker.stdin.write(payload)
            worker.stdin.close()

        while any(worker.poll() is None for worker in workers):
            return_codes = [worker.poll() for worker in workers]
            if any(code is not None and code != 0 for code in return_codes):
                raise RuntimeError(f"DDP worker failure: return codes {return_codes}")
            time.sleep(0.1)
        return_codes = [worker.returncode for worker in workers]
    except BaseException:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        raise

    if any(return_code != 0 for return_code in return_codes):
        raise RuntimeError(f"DDP worker failure: return codes {return_codes}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return pl.DataFrame(payload["history"]), payload["latest"]
    return None, None




