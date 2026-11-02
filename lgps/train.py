import contextlib
from dataclasses import fields, is_dataclass
import importlib
import inspect
import json
import math
import os
import random
import socket
import time
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from pathlib import Path
from .config import LGPSRuntimeConfig, OUTPUT_DIR, normalize_lgps_train_config
from .targets import bucket_edges, collapse_probabilities, open_horizon
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from .dataset import LGPSDataset, cleanup_training_cache
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

class WandbAdapter:
    # Optional Weights & Biases reporting adapter.
    def __init__(self, project, entity=None, name=None, mode=None, config=None):
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as error:
            raise ImportError("W&B reporting requires `pip install wandb`") from error
        self.wandb = wandb
        self.run = wandb.init(
            project=project, entity=entity, name=name, mode=mode,
            config=config or {},
        )

    def log_round_metrics(self, round_index, loss, calibration_error):
        self.run.log(
            {
                'val/loss/goal': float(loss),
                'val/calibration_error/goal': float(calibration_error),
            },
            step=int(round_index),
            commit=True,
        )

    @staticmethod
    def _config_value(value):
        if is_dataclass(value):
            return {
                item.name: WandbAdapter._config_value(getattr(value, item.name))
                for item in fields(value)
            }
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): WandbAdapter._config_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [WandbAdapter._config_value(item) for item in value]
        return value

    @classmethod
    def lgps_config(cls, model_config, train_config, runtime_config, *,
                    batch_size_per_gpu, accumulation_steps, effective_batch_size):
        return {
            'model': cls._config_value(model_config),
            'train': cls._config_value(train_config),
            'runtime': cls._config_value(runtime_config),
            'derived': {
                'num_outputs': model_config.num_outputs,
                'batch_size_per_gpu': int(batch_size_per_gpu),
                'physical_global_batch_size': int(batch_size_per_gpu * runtime_config.world_size),
                'gradient_accumulation_steps': int(accumulation_steps),
                'effective_global_batch_size': int(effective_batch_size),
            },
        }

    def log_validation(self, metrics, epoch, probabilities=None, targets=None,
                       horizon_frame_rate=20):
        payload = {}

        # ROC and PR are curves over decision thresholds, not AUC trends over epochs.
        # Log W&B's native curve plots from this epoch's predictions.
        if probabilities is not None and targets is not None and len(targets):
            class_names = ('blue', 'orange', 'neither')
            roc_auc_values = [metrics.get(f'validation_{name}_roc_auc') for name in class_names]
            pr_auc_values = [metrics.get(f'validation_{name}_pr_auc') for name in class_names]
            if any(value is not None for value in roc_auc_values):
                payload['val/roc_curve'] = self.wandb.plot.roc_curve(
                    y_true=targets,
                    y_probas=probabilities,
                    labels=list(class_names),
                    title='Goal ROC Curve',
                )
            if any(value is not None for value in pr_auc_values):
                payload['val/pr_curve'] = self.wandb.plot.pr_curve(
                    y_true=targets,
                    y_probas=probabilities,
                    labels=list(class_names),
                    title='Goal Precision-Recall Curve',
                )

        horizon_values = []
        horizon_frames = []
        for second in range(1, 31):
            value = metrics.get(f'validation_accuracy_horizon_{second}s')
            if value is not None:
                horizon_frames.append(second * horizon_frame_rate)
                horizon_values.append(float(value))
        if horizon_values:
            payload['val/accuracy'] = self.wandb.plot.line_series(
                xs=horizon_frames,
                ys=[horizon_values],
                keys=['goal'],
                title='Goal Accuracy by Frames Until Goal',
                xname='Frames Until Goal',
            )

        # Drop an absent custom plot instead of asking W&B to serialize None.
        payload = {key: value for key, value in payload.items() if value is not None}
        self.run.log(payload)

    def log_checkpoint_plots(self, output_dir, epoch):
        files = {
            'roc_auc': 'lgps_roc_auc.png',
            'probability_distribution': 'lgps_probability_distribution.png',
            'pr_auc': 'lgps_pr_auc.png',
            'calibration': 'lgps_calibration.png',
        }
        payload = {}
        for name, filename in files.items():
            path = Path(output_dir) / filename
            if path.is_file():
                payload[f'val/checkpoint/{name}_plot'] = self.wandb.Image(
                    str(path), caption=f'Validation checkpoint, epoch {epoch}'
                )
        if len(payload) > 1:
            self.run.log(payload)

    def log_test(self, metrics, output_dir, metrics_path):
        payload = {}
        files = {
            'roc_auc': 'lgps_roc_auc.png',
            'pr_auc': 'lgps_pr_auc.png',
            'calibration': 'lgps_calibration.png',
            'probability_distribution': 'lgps_probability_distribution.png',
        }
        for name, filename in files.items():
            path = Path(output_dir) / filename
            if path.is_file():
                payload[f'test/{name}_plot'] = self.wandb.Image(
                    str(path), caption=f'Test {name.replace("_", " ")}'
                )
        if payload:
            self.run.log(payload)
        if Path(metrics_path).is_file():
            artifact = self.wandb.Artifact('lgps-test-metrics', type='evaluation')
            artifact.add_file(str(metrics_path), name='test/metrics.json')
            self.run.log_artifact(artifact, aliases=['latest'])

    def finish(self):
        self.run.finish()


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


class LGPSClassificationMetrics:
    def __init__(self, device, targets_per_team, store_predictions=False, horizon_seconds=10.0,
                 geometric_width=1.0, horizon_max=30.0, two_way_prediction=False):
        self.device = device
        self.targets_per_team = targets_per_team
        self.horizon_seconds = None if open_horizon(horizon_seconds) else horizon_seconds
        self.store_predictions = store_predictions
        self.geometric_width = geometric_width
        self.horizon_max = float(horizon_max)
        self.horizon_steps = int(math.floor(self.horizon_max))
        self.two_way_prediction = bool(two_way_prediction)

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
        if self.horizon_seconds is None:
            self.horizon_correct = torch.zeros(self.horizon_steps, device=device, dtype=torch.float64)
            self.horizon_count = torch.zeros(self.horizon_steps, device=device, dtype=torch.float64)
        self.probabilities = []
        self.targets = []

    @torch.no_grad()
    def update(self, loss, logits, targets, horizon_targets=None):
        probabilities = collapse_probabilities(
            logits.detach().float().softmax(dim=-1), self.targets_per_team, self.horizon_seconds,
            self.two_way_prediction,
        )
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

        if self.horizon_seconds is None:
            if horizon_targets is None:
                raise ValueError("open-horizon metrics require per-horizon targets")
            team_width = self.targets_per_team
            full = logits.detach().float().softmax(dim=-1)
            team_bucket_probabilities = full[:, :2 * team_width].reshape(n, 2, team_width)
            edges = torch.as_tensor(
                bucket_edges(self.targets_per_team, self.geometric_width, None, self.horizon_max),
                device=self.device,
                dtype=team_bucket_probabilities.dtype,
            )
            horizon_count = min(self.horizon_steps, horizon_targets.shape[1])
            horizons = torch.arange(1, horizon_count + 1, device=self.device, dtype=edges.dtype)
            regular_fraction = (
                (horizons[:, None] - edges[:-1])
                / (edges[1:] - edges[:-1])
            ).clamp(0.0, 1.0)
            # The final per-team bucket is open ended. Its mass has no
            # resolution within horizon_max, so it stays outside finite-horizon mass.
            bucket_fraction = torch.cat(
                (regular_fraction, torch.zeros((horizon_count, 1), device=self.device, dtype=edges.dtype)),
                dim=1,
            )
            team_by_horizon = torch.einsum(
                "nbk,hk->nhb", team_bucket_probabilities, bucket_fraction
            )
            neither_by_horizon = 1.0 - team_by_horizon.sum(dim=-1, keepdim=True)
            horizon_probabilities = torch.cat((team_by_horizon, neither_by_horizon), dim=-1)
            horizon_predictions = horizon_probabilities.argmax(dim=-1)
            self.horizon_correct[:horizon_count] += (
                horizon_predictions == horizon_targets[:, :horizon_count].long()
            ).sum(dim=0).double()
            self.horizon_count[:horizon_count] += n

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

        if self.horizon_seconds is None:
            totals = torch.cat((totals, self.horizon_correct, self.horizon_count))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                totals,
                op=dist.ReduceOp.SUM,
            )

        loss_sum, correct, collapsed_three_way_loss_sum, brier_sum, count = totals[:5]

        if count.item() == 0:
            raise RuntimeError("The data split produced zero usable rows.")

        result = {
            "loss": (loss_sum / count).item(),
            "collapsed_three_way_loss": (collapsed_three_way_loss_sum / count).item(),
            "accuracy": (correct / count).item(),
            "brier": (brier_sum / count).item(),
            "examples": int(count.item()),
        }
        if self.horizon_seconds is None:
            correct_by_horizon = totals[5:5 + self.horizon_steps]
            count_by_horizon = totals[5 + self.horizon_steps:5 + 2 * self.horizon_steps]
            for index in range(self.horizon_steps):
                result[f"accuracy_horizon_{index + 1}s"] = (
                    correct_by_horizon[index] / count_by_horizon[index]
                ).item()
        return result


class LGPSTrainingMetrics:
    # Accumulate inexpensive metrics during training.

    def __init__(self, device, targets_per_team, horizon_seconds=10.0,
                 two_way_prediction=False):
        self.targets_per_team = targets_per_team
        self.horizon_seconds = horizon_seconds
        self.two_way_prediction = bool(two_way_prediction)
        self.loss = torch.zeros((), device=device, dtype=torch.float64)
        self.brier = torch.zeros((), device=device, dtype=torch.float64)
        self.examples = torch.zeros((), device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, loss, logits, targets):
        probabilities = collapse_probabilities(
            logits.detach().float().softmax(dim=-1), self.targets_per_team, self.horizon_seconds,
            self.two_way_prediction,
        )
        count = targets.shape[0]
        self.loss += loss.detach().double() * count
        target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
        self.brier += (probabilities.square().sum(dim=-1) - 2 * target_probability + 1).sum().double()
        self.examples += count

    def compute(self):
        totals = torch.stack((self.loss, self.brier, self.examples))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss, brier, examples = totals
        if not examples.item():
            raise RuntimeError("no training examples were observed")
        result = {
            "loss": (loss / examples).item(),
            "brier": (brier / examples).item(),
        }
        return result


class LGPSResults:
    CLASS_NAMES = ("blue", "orange", "neither")

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_losses = []
        self.history = []
        self.round_metrics = []
        self.final_metrics = {}

    def add_round_metrics(self, round_index, loss, calibration_error):
        self.round_metrics.append({
            'round': int(round_index),
            'round_loss': float(loss),
            'round_calibration_error': float(calibration_error),
        })

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
    def _reported_metrics(metrics):
        return {
            key: value for key, value in metrics.items()
            if not key.startswith("train_")
        }

    @classmethod
    def _metrics_payload(cls, latest, history):
        return {
            "latest": cls._reported_metrics(latest),
            "history": [cls._reported_metrics(row) for row in history],
        }

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
            ylabel="Combined training loss",
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

        metrics_payload = self._metrics_payload(metrics, self.history)
        metrics_payload['rounds'] = self.round_metrics
        (self.output_dir / "metrics.json").write_text(
            json.dumps(self._json_value(metrics_payload), indent=2),
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
        config = normalize_lgps_train_config(config)
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
        self.training_step = 0
        self.wandb_adapter = (
            WandbAdapter(self.runtime_config.wandb_project, self.runtime_config.wandb_entity,
                         self.runtime_config.wandb_run_name, self.runtime_config.wandb_mode,
                         config=WandbAdapter.lgps_config(
                             model_config, config, self.runtime_config,
                             batch_size_per_gpu=self.batch_size,
                             accumulation_steps=self.accumulation_steps,
                             effective_batch_size=self.effective_batch_size,
                         ))
            if self.runtime_config.wandb_project is not None and context.is_main else None
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
        correction = self.effective_batch_size / examples
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)

        if self.scaler.is_enabled():
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

    @torch.no_grad()
    def batch_loss_and_calibration_error(self, loss, logits, targets):
        probabilities = collapse_probabilities(
            logits.detach().float().softmax(dim=-1),
            self.model_config.targets_per_team,
            self.model_config.horizon_seconds,
            self.model_config.two_way_prediction,
        )
        confidence, prediction = probabilities.max(dim=-1)
        correct = (prediction == targets).to(torch.float64)
        confidence = confidence.to(torch.float64)
        bin_count = 15
        bin_index = (confidence * bin_count).long().clamp_max(bin_count - 1)
        stats = torch.zeros((bin_count, 3), dtype=torch.float64, device=self.device)
        ones = torch.ones_like(confidence)
        stats[:, 0].scatter_add_(0, bin_index, ones)
        stats[:, 1].scatter_add_(0, bin_index, confidence)
        stats[:, 2].scatter_add_(0, bin_index, correct)
        weighted_loss = torch.stack((loss.detach().double() * targets.numel(),
                                     torch.tensor(float(targets.numel()), device=self.device, dtype=torch.float64)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(weighted_loss, op=dist.ReduceOp.SUM)
        active = stats[:, 0] > 0
        calibration_error = (
            (stats[active, 0] / stats[:, 0].sum())
            * (stats[active, 2] / stats[active, 0] - stats[active, 1] / stats[active, 0]).abs()
        ).sum()
        return (weighted_loss[0] / weighted_loss[1]).item(), calibration_error.item()

    def training_batches(self):
        iterator = iter(BatchPrefetcher(self.train_loader, self.device))

        def next_batch():
            batch = next(iterator, None)
            local_count = 0 if batch is None else batch["temporal_targets"].shape[0]
            if local_count == 0:
                return None
            return batch, local_count

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
                batch, local_count = current
                global_count = torch.tensor(local_count, device=self.device, dtype=torch.int64)
                if self.context.is_distributed:
                    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
                yield (batch, local_count, int(global_count.item()),
                       group_index + 1 == shared_available)
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
                camera_features=batch.get('camera_features'),
                camera_visible=batch.get('camera_visible'),
                camera_valid=batch.get('camera_valid'),
                camera_age=batch.get('camera_age'),
                camera_relation=batch.get('camera_relation'),
            )
        if (not evaluation and model_override is None and model.training
                and self.config.activation_checkpointing):
            return checkpoint(call_model, batch["ball"], use_reentrant=False)
        return call_model(batch["ball"])
    def train_epoch(self, epoch):
        self.model.train()
        metrics = LGPSTrainingMetrics(
            self.device, self.model_config.targets_per_team, self.model_config.horizon_seconds,
            self.model_config.two_way_prediction,
        )
        start_time = time.perf_counter()
        last_log_time = start_time
        local_examples = 0
        self.context.print(f"Epoch {epoch}/{self.config.epochs} | training started")
        self.zero_grad()
        gradient_norm = torch.zeros((), device=self.device)
        accumulated_examples = 0
        for batch_index, (batch, local_count, global_count, is_last) in enumerate(
            self.training_batches(), start=1,
        ):
            if batch["ball"].device != self.device:
                batch = BatchDevice.move(batch, self.device)
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
                    logits = self.forward(batch)
                    loss = -(batch["temporal_targets"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                    scaled_loss = loss * (local_count * self.context.world_size / self.effective_batch_size)
                self.scaler.scale(scaled_loss).backward()
            if is_last:
                gradient_norm = self.optimizer_step(accumulated_examples)
                accumulated_examples = 0
            metrics.update(loss, logits, batch["collapsed_targets"])
            local_examples += local_count
            self.training_step += 1
            round_loss, round_calibration_error = self.batch_loss_and_calibration_error(
                loss, logits, batch['collapsed_targets']
            )
            if self.context.is_main:
                self.results.add_round_metrics(
                    self.training_step, round_loss, round_calibration_error
                )
                if self.wandb_adapter is not None:
                    self.wandb_adapter.log_round_metrics(
                        self.training_step, round_loss, round_calibration_error
                    )
            if self.context.is_main and batch_index % self.config.log_interval == 0:
                now = time.perf_counter()
                elapsed = now - start_time
                interval = now - last_log_time
                throughput = local_examples * self.context.world_size / elapsed if elapsed > 0 else 0.0
                print(
                    f"Epoch {epoch}/{self.config.epochs} | batch {batch_index:,} | "
                    f"loss {loss.detach().item():.5f} | goal {loss.detach().item():.5f} | "
                    f"grad {float(gradient_norm):.4f} | lr {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"local examples {local_examples:,} | {throughput:,.1f} global examples/s | "
                    f"{interval:.1f}s since last log"
                )
                last_log_time = now
        if accumulated_examples:
            gradient_norm = self.optimizer_step(accumulated_examples)
        result = metrics.compute()
        elapsed = time.perf_counter() - start_time
        self.context.print(
            f"Epoch {epoch}/{self.config.epochs} | training finished | "
            f"loss {result['loss']:.5f} | brier {result['brier']:.5f} | time {elapsed:.1f}s"
        )
        return result
    @torch.inference_mode()
    def evaluate(self, epoch, loader, label, store_predictions):
        self.model.eval()
        metrics = LGPSClassificationMetrics(
            self.device, self.model_config.targets_per_team,
            store_predictions=store_predictions,
            horizon_seconds=self.model_config.horizon_seconds,
            geometric_width=self.model_config.geometric_width,
            horizon_max=self.model_config.horizon_max,
            two_way_prediction=self.model_config.two_way_prediction,
        )
        start_time = time.perf_counter()
        self.context.print(f"Epoch {epoch}/{self.config.epochs} | {label} started")
        for batch_index, batch in enumerate(loader, start=1):
            batch = BatchDevice.move(batch, self.device)
            with torch.autocast(
                device_type=self.device.type, dtype=self.autocast_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.forward(batch, evaluation=True)
                loss = -(batch["temporal_targets"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            metrics.update(loss, logits, batch["collapsed_targets"], batch.get("horizon_targets"))
            if self.context.is_main and batch_index % self.config.log_interval == 0:
                print(f"Epoch {epoch}/{self.config.epochs} | {label} batch {batch_index:,}")
        result = metrics.compute()
        elapsed = time.perf_counter() - start_time
        rate = result["examples"] / elapsed if elapsed > 0 else 0.0
        self.context.print(
            f"Epoch {epoch}/{self.config.epochs} | {label} finished | "
            f"loss {result['loss']:.5f} | accuracy {result['accuracy']:.4f} | "
            f"collapsed three-way loss {result['collapsed_three_way_loss']:.5f} | "
            f"brier {result['brier']:.5f} | examples {result['examples']:,} | "
            f"time {elapsed:.1f}s | throughput {rate / self.context.world_size:,.1f}/device "
            f"({rate:,.1f} global/s)"
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

        if self.wandb_adapter is not None:
            self.wandb_adapter.finish()

        self.context.print(
            "Training complete"
        )

        if self.context.is_main:
            return pl.DataFrame(history)

        return None

def create_model(model_config, train_config):
    train_config = normalize_lgps_train_config(train_config)
    return LGPSTransformer(
        config=model_config,
        player_view=train_config.player_view is not None,
        camera_encoder_settings=train_config.player_view,
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
        # Newer PyTorch releases batch gradient copies into each ready bucket.
        # Keep training compatible with older PyTorch DDP signatures.
        if "batched_grad_copy" in inspect.signature(DDP).parameters:
            ddp_kwargs["batched_grad_copy"] = True
        model = DDP(model, **ddp_kwargs)
    return model


def build_trainer(model_config, train_config, context, runtime_config=None):
    train_config = normalize_lgps_train_config(train_config)
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
