import importlib
import json
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from dataclasses import fields, is_dataclass
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from .targets import collapse_probabilities, open_horizon

plt.switch_backend("Agg")

@torch.inference_mode()
def evaluate_model(trainer, epoch, loader, label, store_predictions):
    trainer.model.eval()
    metrics = LGPSClassificationMetrics(
        trainer.device, trainer.model_config.targets_per_team,
        store_predictions=store_predictions,
        horizon_seconds=trainer.model_config.horizon_seconds,
        two_way_prediction=trainer.model_config.two_way_prediction,
    )
    start_time = time.perf_counter()
    trainer.context.print(f"Epoch {epoch}/{trainer.config.epochs} | {label} started")

    for batch_index, batch in enumerate(loader, start=1):
        batch = {key: value.to(trainer.device, non_blocking=True) for key, value in batch.items()}

        with torch.autocast(
            device_type=trainer.device.type, dtype=trainer.autocast_dtype,
            enabled=trainer.amp_enabled,
        ):
            logits = trainer.forward(batch, evaluation=True)
            loss = -(batch["temporal_targets"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        metrics.update(loss, logits, batch["collapsed_targets"])

        if trainer.context.is_main and batch_index % trainer.config.log_interval == 0:
            print(f"Epoch {epoch}/{trainer.config.epochs} | {label} batch {batch_index:,}")

    result = metrics.compute()
    elapsed = time.perf_counter() - start_time
    examples = result.pop("examples")
    rate = examples / elapsed if elapsed > 0 else 0.0
    examples_label = f"examples {examples:,} | " if label != "validation" else ""
    trainer.context.print(
        f"Epoch {epoch}/{trainer.config.epochs} | {label} finished | "
        f"loss {result['loss']:.5f} | accuracy {result['accuracy']:.4f} | "
        f"collapsed loss {result['collapsed_loss']:.5f} | "
        f"brier {result['brier']:.5f} | {examples_label}"
        f"time {elapsed:.1f}s | throughput {rate / trainer.context.world_size:,.1f}/device "
        f"({rate:,.1f} global/s)"
    )
    return result, metrics.arrays()


class WandbAdapter:
    # Optional Weights & Biases reporting adapter.
    TRAIN_BATCH_METRICS = ('loss', 'loss/goal', 'throughput')

    def __init__(self, project, entity=None, name=None, mode=None, config=None,
                 run_id=None):
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as error:
            raise ImportError("W&B reporting requires `pip install wandb`") from error

        self.wandb = wandb
        self.run = wandb.init(
            project=project, entity=entity, name=name, mode=mode,
            config=config or {}, id=run_id,
            resume="allow" if run_id else None,
        )
        self.run.define_metric('val/*', step_metric='epoch')

        for metric in self.TRAIN_BATCH_METRICS:
            self.run.define_metric(f'train/{metric}', step_metric='train/batch')

    def log_training(self, metrics, batch, epoch):
        payload = {'train/batch': int(batch), 'train/epoch': int(epoch)}
        payload.update({f'train/{key}': value for key, value in metrics.items()})
        self.run.log(payload)

    def configure_auxiliary_plots(self, loss_task_names, weight_task_names):
        # W&B's scalar panels use the training batch index as their x axis.
        for task_name in loss_task_names:
            self.run.define_metric(f'train/loss/{task_name}', step_metric='train/batch')

        for task_name in weight_task_names:
            self.run.define_metric(f'train/weight/{task_name}', step_metric='train/batch')

    @staticmethod
    def _config_value(value):
        if isinstance(value, type):
            return f'{value.__module__}.{value.__qualname__}'

        task_type = getattr(value, 'task_type', None)

        if task_type is not None and is_dataclass(value):
            return {
                'task': f'{task_type.__module__}.{task_type.__qualname__}',
                **{
                    item.name: WandbAdapter._config_value(getattr(value, item.name))
                    for item in fields(value)
                },
            }

        if is_dataclass(value):
            return {
                item.name: WandbAdapter._config_value(getattr(value, item.name))
                for item in fields(value)
            }

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                (f'{key.__module__}.{key.__qualname__}' if isinstance(key, type) else str(key)):
                WandbAdapter._config_value(item)
                for key, item in value.items()
            }

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

    def log_validation(self, metrics, epoch, probabilities=None, targets=None):
        payload = {'epoch': int(epoch)}

        for name, value in metrics.items():
            if (
                not name.startswith('validation_')
                or not isinstance(value, (int, float, np.integer, np.floating))
            ):
                continue

            value = float(value)

            if math.isfinite(value):
                payload[f"val/{name.removeprefix('validation_')}"] = value

        for name in ('learning_rate', 'muon_learning_rate'):
            value = metrics.get(name)

            if isinstance(value, (int, float, np.integer, np.floating)):
                value = float(value)

                if math.isfinite(value):
                    payload[f'val/{name}'] = value

        self.run.log({key: value for key, value in payload.items() if value is not None})

    def log_checkpoint_plots(self, output_dir, epoch):
        files = {
            'epoch_losses': 'epoch_losses.png',
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

        if payload:
            self.run.log({'epoch': int(epoch), **payload})

    def log_test(self, metrics, output_dir, metrics_path):
        files = {
            'roc_auc': 'lgps_roc_auc.png',
            'pr_auc': 'lgps_pr_auc.png',
            'calibration': 'lgps_calibration.png',
            'probability_distribution': 'lgps_probability_distribution.png',
        }
        payload = {}

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


class LGPSClassificationMetrics:
    def __init__(self, device, targets_per_team, store_predictions=False, horizon_seconds=10.0,
                 two_way_prediction=False):
        self.device = device
        self.targets_per_team = targets_per_team
        self.horizon_seconds = None if open_horizon(horizon_seconds) else horizon_seconds
        self.store_predictions = store_predictions
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
        self.collapsed_loss_sum = torch.zeros(
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
    def update(self, loss, logits, targets):
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

        self.collapsed_loss_sum += F.nll_loss(
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
                self.collapsed_loss_sum,
                self.brier_sum,
                self.count,
            )
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                totals,
                op=dist.ReduceOp.SUM,
            )

        loss_sum, correct, collapsed_loss_sum, brier_sum, count = totals[:5]

        if count.item() == 0:
            raise RuntimeError("The data split produced zero usable rows.")

        result = {
            "loss": (loss_sum / count).item(),
            "collapsed_loss": (collapsed_loss_sum / count).item(),
            "accuracy": (correct / count).item(),
            "brier": (brier_sum / count).item(),
            "examples": int(count.item()),
        }
        return result


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

    @staticmethod
    def _calibration_error(probabilities, targets, bins=15):
        if probabilities.size == 0:
            return None

        confidence = probabilities.max(axis=1)
        prediction = probabilities.argmax(axis=1)
        correct = (prediction == targets).astype(np.float64)
        bin_index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
        calibration_error = 0.0

        for index in range(bins):
            selected = bin_index == index

            if selected.any():
                calibration_error += selected.mean() * abs(
                    correct[selected].mean() - confidence[selected].mean()
                )

        return float(calibration_error)

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

        if label == "validation":
            metrics["validation_calibration_error"] = self._calibration_error(
                probabilities, targets,
            )

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
        (self.output_dir / "metrics.json").write_text(
            json.dumps(self._json_value(metrics_payload), indent=2),
            encoding="utf-8",
        )

        return metrics
