import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from props import BALL_FEATURES, PLAYER_FEATURES, PLAYER_SLOTS


def _masked_mean(values, valid):
    # A missing future state must contribute neither loss nor a NaN gradient.
    count = valid.sum()
    loss = torch.where(valid, values, torch.zeros_like(values)).sum() / count.clamp_min(1)
    return loss, bool(count.detach().item())


def _state_loss(predicted, target, scale, binary_indices=()):
    # Continuous fields use normalized world units; discrete state flags use
    # logits so the head learns pressed/not-pressed and present/not-present.
    normalized = target.float() / scale.to(target.device)
    feature_count = target.shape[-1]
    quaternion_indices = tuple(range(9, 13))
    continuous_indices = tuple(index for index in range(feature_count)
                               if index not in quaternion_indices and index not in binary_indices)
    parts = []
    active = False

    if continuous_indices:
        observed = torch.isfinite(normalized[..., continuous_indices])
        error = F.smooth_l1_loss(predicted.float()[..., continuous_indices],
                                 torch.nan_to_num(normalized[..., continuous_indices]), reduction='none')
        value, present = _masked_mean(error, observed)
        parts.append(value)
        active |= present

    if binary_indices:
        observed = torch.isfinite(target[..., binary_indices])
        error = F.binary_cross_entropy_with_logits(
            predicted.float()[..., binary_indices],
            torch.nan_to_num(target.float()[..., binary_indices]).clamp(0, 1),
            reduction='none',
        )
        value, present = _masked_mean(error, observed)
        parts.append(value)
        active |= present

    observed = torch.isfinite(target[..., 9:13]).all(-1)
    predicted_quaternion = F.normalize(predicted.float()[..., 9:13], dim=-1)
    target_quaternion = F.normalize(torch.nan_to_num(target.float()[..., 9:13]), dim=-1)
    rotation_error = 1.0 - (predicted_quaternion * target_quaternion).sum(-1).abs().clamp(max=1.0)
    value, present = _masked_mean(rotation_error, observed)
    parts.append(value)
    active |= present

    return sum(parts), active


def _future_rows(frame, indices, future_frames, sample_rate):
    # Match future rows by physical time within one replay stint. The sample
    # rate belongs to the model config; gaps in filtered rows stay invalid.
    replay = frame['replay_id'].to_list()
    stint = frame['_window_id'].to_list()
    times = frame['_source_time'].to_numpy()
    groups = {}
    for row, key in enumerate(zip(replay, stint)):
        groups.setdefault(key, []).append(row)
    groups = {key: np.asarray(rows, dtype=np.int64) for key, rows in groups.items()}
    future = np.full((len(indices), len(future_frames)), -1, dtype=np.int64)

    for output_row, row in enumerate(indices):
        rows = groups[(replay[row], stint[row])]
        group_times = times[rows]
        for horizon, offset in enumerate(future_frames):
            target_time = times[row] + offset / sample_rate
            position = np.searchsorted(group_times, target_time)
            if position < len(rows) and abs(group_times[position] - target_time) <= 0.5 / sample_rate:
                future[output_row, horizon] = rows[position]

    return future


def build_aux_targets(frame, indices, future_frames, aux_tasks, sample_rate):
    indices = np.asarray(indices, dtype=np.int64)
    future_frames = tuple(() if future_frames is None else future_frames)
    result = {}
    if future_frames and (LGPSBallStateAux in aux_tasks or LGPSPlayerStateAux in aux_tasks):
        future = _future_rows(frame, indices, future_frames, sample_rate)
        selected = np.maximum(future, 0)
        valid = future >= 0
        if LGPSBallStateAux in aux_tasks:
            ball = frame.select(BALL_FEATURES).to_numpy().astype(np.float32)
            ball_targets = ball[selected].copy()
            ball_targets[~valid] = np.nan
            result['aux_ball_targets'] = torch.from_numpy(ball_targets)
        if LGPSPlayerStateAux in aux_tasks:
            players = np.stack([
                frame.select(tuple(f'player_{slot}_{name}' for name in PLAYER_FEATURES))
                .to_numpy().astype(np.float32)
                for slot in PLAYER_SLOTS
            ], axis=1)
            player_targets = players[selected].copy()
            player_targets[~valid] = np.nan
            result['aux_player_targets'] = torch.from_numpy(player_targets)
    if LGPSNextTouchAux in aux_tasks:
        result['aux_next_touch_slot'] = torch.from_numpy(
            frame['_next_touch_slot'].to_numpy()[indices].astype(np.int64)
        )
        result['aux_next_touch_seconds'] = torch.from_numpy(
            frame['_next_touch_seconds'].to_numpy()[indices].astype(np.float32)
        )
    return result


class LGPSBallStateAux(nn.Module):
    def __init__(self, d_model, future_frames):
        super().__init__()
        self.future_frames = tuple(() if future_frames is None else future_frames)
        self.heads = nn.ModuleDict({
            str(frame): nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                      nn.GELU(), nn.Linear(d_model, len(BALL_FEATURES)))
            for frame in self.future_frames
        })

    def forward(self, pooled):
        return {f'ball_{frame}': self.heads[str(frame)](pooled)
                for frame in self.future_frames}

    @staticmethod
    def decode(predicted):
        from .dataset import LGPSScaler

        # The regression head uses scaled physical units; restore replay
        # units and place its quaternion back on the unit sphere.
        state = predicted.float() * LGPSScaler.BALL_SCALE.to(predicted.device)
        state[..., 9:13] = F.normalize(state[..., 9:13], dim=-1)
        return state

    def losses(self, predictions, targets):
        from .dataset import LGPSScaler

        return {
            f'ball_{frame}': _state_loss(predictions[f'ball_{frame}'],
                                        targets[:, index], LGPSScaler.BALL_SCALE)
            for index, frame in enumerate(self.future_frames)
        }


class LGPSPlayerStateAux(nn.Module):
    def __init__(self, d_model, future_frames):
        super().__init__()
        self.future_frames = tuple(() if future_frames is None else future_frames)
        self.heads = nn.ModuleDict({
            str(frame): nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model),
                                      nn.GELU(), nn.Linear(d_model, len(PLAYER_FEATURES)))
            for frame in self.future_frames
        })
        self.binary_indices = tuple(PLAYER_FEATURES.index(name) for name in (
            'demoed', 'on_field', 'handbrake', 'is_boosting', 'supersonic',
            'has_dodge', 'is_jumping', 'is_double_jumping', 'is_dodging',
        ))

    def forward(self, player_streams, pooled):
        features = torch.cat((player_streams, pooled[:, None].expand_as(player_streams)), dim=-1)
        return {f'player_{frame}': self.heads[str(frame)](features)
                for frame in self.future_frames}

    def decode(self, predicted):
        from .dataset import LGPSScaler

        state = predicted.float() * LGPSScaler.PLAYER_SCALE.to(predicted.device)
        state[..., 9:13] = F.normalize(state[..., 9:13], dim=-1)
        state[..., self.binary_indices] = (predicted[..., self.binary_indices] >= 0).float()
        return state

    def losses(self, predictions, targets):
        from .dataset import LGPSScaler

        return {
            f'player_{frame}': _state_loss(predictions[f'player_{frame}'],
                                          targets[:, index], LGPSScaler.PLAYER_SCALE,
                                          self.binary_indices)
            for index, frame in enumerate(self.future_frames)
        }


class LGPSNextTouchAux(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.player_score = nn.Linear(d_model, 1)
        self.global_score = nn.Linear(d_model, d_model)
        self.player_time = nn.Linear(d_model, 1)

    def forward(self, player_streams, pooled):
        # A shared scorer over player tokens keeps touch identity equivariant
        # when players of the same team exchange input slots.
        conditioned = player_streams + self.global_score(pooled)[:, None]
        logits = self.player_score(conditioned).squeeze(-1)
        seconds = F.softplus(self.player_time(conditioned).squeeze(-1))
        return logits, seconds

    def loss(self, predictions, slot, seconds):
        logits, predicted_seconds = predictions
        valid = (slot >= 0) & (slot < logits.shape[-1]) & torch.isfinite(seconds) & (seconds > 0)
        if not bool(valid.any().item()):
            return (logits.sum() + predicted_seconds.sum()) * 0.0, False
        identity = F.cross_entropy(logits[valid].float(), slot[valid])
        selected_time = predicted_seconds[valid].gather(1, slot[valid, None]).squeeze(1)
        timing = F.smooth_l1_loss(torch.log1p(selected_time.float()), torch.log1p(seconds[valid].float()))
        return identity + timing, True


class LGPSGradNorm:
    def __init__(self, task_names, alpha=1.5, momentum=0.9, device='cpu'):
        self.task_names = tuple(task_names)
        self.alpha = float(alpha)
        self.momentum = float(momentum)
        self.initial = torch.full((len(task_names) + 1,), float('nan'), device=device)
        self.weights = torch.ones(len(task_names), device=device)

    def combine(self, primary, auxiliary, shared):
        # Match the *combined* auxiliary gradient budget to the primary task.
        # Gradients are measured at the last shared entity representation, so
        # task-head parameters do not distort the comparison.
        losses = [primary] + [auxiliary[name][0] for name in self.task_names]
        active = torch.tensor([True] + [auxiliary[name][1] for name in self.task_names],
                              device=shared.device, dtype=torch.int32)
        norms = []
        for loss in losses:
            gradient = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)[0]
            norms.append(torch.linalg.vector_norm(gradient.float()) if gradient is not None
                         else shared.new_zeros((), dtype=torch.float32))
        norms = torch.stack(norms).detach()
        values = torch.stack([loss.detach().float() for loss in losses])

        if dist.is_available() and dist.is_initialized():
            # Every rank must use the same task weights before DDP averages
            # the actual model gradients at the accumulation boundary.
            dist.all_reduce(norms)
            dist.all_reduce(values)
            dist.all_reduce(active, op=dist.ReduceOp.MAX)
            norms /= dist.get_world_size()
            values /= dist.get_world_size()

        with torch.no_grad():
            active = active.bool()
            first = active & ~torch.isfinite(self.initial)
            self.initial[first] = values[first].clamp_min(1e-8)
            rates = values / self.initial.clamp_min(1e-8)
            relative = (rates[1:] / rates[0].clamp_min(1e-8)).clamp(0.1, 10.0)
            active_aux = active[1:]
            if active_aux.any():
                target_norm = norms[0] * relative.pow(self.alpha) / active_aux.sum()
                proposed = (target_norm / norms[1:].clamp_min(1e-8)).clamp(0.01, 10.0)
                self.weights[active_aux] = (
                    self.momentum * self.weights[active_aux]
                    + (1.0 - self.momentum) * proposed[active_aux]
                )

        combined = primary + sum(
            self.weights[index].detach() * auxiliary[name][0]
            for index, name in enumerate(self.task_names)
        )
        return combined, {name: float(self.weights[index].item())
                          for index, name in enumerate(self.task_names)}
