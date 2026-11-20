import numpy as np
from dataclasses import dataclass
from typing import Literal
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from shared import BALL_FEATURES, PLAYER_FEATURES, PLAYER_SLOTS


def _masked_mean(values, valid):
    # A missing future state must contribute neither loss nor a NaN gradient.
    count = valid.sum()
    global_count = count.detach().clone()
    world_size = 1

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()

    loss = torch.where(valid, values, torch.zeros_like(values)).sum() * world_size / global_count.clamp_min(1)
    return loss, bool(count.detach().item())


def _state_loss(predicted, target, scale, binary_indices=(), *, beta=1.0,
                continuous_weight=1.0, binary_weight=1.0, rotation_weight=1.0):
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
                                 torch.nan_to_num(normalized[..., continuous_indices]),
                                 beta=beta, reduction='none')
        value, present = _masked_mean(error, observed)
        parts.append(continuous_weight * value)
        active |= present

    if binary_indices:
        observed = torch.isfinite(target[..., binary_indices])
        error = F.binary_cross_entropy_with_logits(
            predicted.float()[..., binary_indices],
            torch.nan_to_num(target.float()[..., binary_indices]).clamp(0, 1),
            reduction='none',
        )
        value, present = _masked_mean(error, observed)
        parts.append(binary_weight * value)
        active |= present

    observed = torch.isfinite(target[..., 9:13]).all(-1) & (
        torch.linalg.vector_norm(torch.nan_to_num(target.float()[..., 9:13]), dim=-1) > 1e-6
    )
    predicted_quaternion = F.normalize(predicted.float()[..., 9:13], dim=-1)
    target_quaternion = F.normalize(torch.nan_to_num(target.float()[..., 9:13]), dim=-1)
    rotation_error = 1.0 - (predicted_quaternion * target_quaternion).sum(-1).abs().clamp(max=1.0)
    value, present = _masked_mean(rotation_error, observed)
    parts.append(rotation_weight * value)
    active |= present

    return sum(parts), active


def _activation(name):
    activations = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU}

    try:
        return activations[name.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError('activation must be gelu, relu, or silu') from error


def _mlp(input_dim, output_dim, hidden_dim, num_hidden_layers, dropout, activation):
    hidden_dim = input_dim if hidden_dim is None else int(hidden_dim)

    if hidden_dim < 1 or num_hidden_layers < 1:
        raise ValueError('MLP hidden_dim and num_hidden_layers must be positive')

    if not 0.0 <= dropout < 1.0:
        raise ValueError('auxiliary dropout must lie in [0, 1)')

    activation_type = _activation(activation)
    layers = [nn.LayerNorm(input_dim)]
    width = input_dim

    for _ in range(num_hidden_layers):
        layers.extend((nn.Linear(width, hidden_dim), activation_type()))

        if dropout:
            layers.append(nn.Dropout(dropout))

        width = hidden_dim

    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def _future_rows(frame, indices, future_frames, sample_rate):
    # Match future rows by physical time within one replay stint. Search each
    # contiguous group as a NumPy matrix instead of looping over sampled rows.
    replay = frame['replay_id'].to_numpy()
    stint_column = 'stint_id' if 'stint_id' in frame.columns else '_window_id'
    stint = frame[stint_column].to_numpy()
    times = frame['_source_time'].to_numpy().astype(np.float64, copy=False)
    boundaries = np.r_[0, np.flatnonzero(
        (replay[1:] != replay[:-1]) | (stint[1:] != stint[:-1])
    ) + 1, frame.height]
    group_ids = np.repeat(np.arange(boundaries.size - 1), np.diff(boundaries))
    future = np.full((len(indices), len(future_frames)), -1, dtype=np.int64)
    offsets = np.asarray(future_frames, dtype=np.float64) / float(sample_rate)
    selected_groups = group_ids[indices]

    for group_id in np.unique(selected_groups):
        output_rows = np.flatnonzero(selected_groups == group_id)
        rows = indices[output_rows]
        start, stop = boundaries[group_id:group_id + 2]
        group_times = times[start:stop]
        targets = times[rows, None] + offsets[None, :]
        positions = np.searchsorted(group_times, targets, side='left')
        in_range = positions < group_times.size
        safe_positions = np.minimum(positions, max(group_times.size - 1, 0))
        matched = in_range & (np.abs(group_times[safe_positions] - targets) <= 0.5 / sample_rate)
        future[output_rows] = np.where(matched, start + safe_positions, -1)

    return future


def _active_player_rows(frame):
    active = []

    for slot in PLAYER_SLOTS:
        on_field = frame[f'player_{slot}_on_field'].to_numpy()
        demoed = frame[f'player_{slot}_demoed'].to_numpy()
        active.append(
            np.isfinite(on_field) & (on_field > 0.5)
            & np.isfinite(demoed) & (demoed < 0.5)
        )

    return np.stack(active, axis=1)


def build_aux_targets(frame, indices, aux_tasks, sample_rate):
    indices = np.asarray(indices, dtype=np.int64)
    result = {}
    ball_settings = aux_tasks.get(LGPSBallStateAux)
    ball_frames = tuple(ball_settings.future_frames) if ball_settings is not None else ()

    if ball_frames:
        future = _future_rows(frame, indices, ball_frames, sample_rate)
        selected = np.maximum(future, 0)
        valid = future >= 0
        ball = frame.select(BALL_FEATURES).to_numpy().astype(np.float32)
        ball_targets = ball[selected].copy()
        ball_targets[~valid] = np.nan
        result['aux_ball_targets'] = torch.from_numpy(ball_targets)

    if LGPSNextTouchAux in aux_tasks:
        if '_next_touch_slot' in frame.columns and '_next_touch_seconds' in frame.columns:
            next_slot = frame['_next_touch_slot'].to_numpy().astype(np.int64)
            next_seconds = frame['_next_touch_seconds'].to_numpy().astype(np.float32)

            if '_source_time' in frame.columns:
                source_time = frame['_source_time'].to_numpy().astype(np.float64, copy=False)
                replay = frame['replay_id'].to_numpy()
                stint_column = 'stint_id' if 'stint_id' in frame.columns else '_window_id'
                stints = frame[stint_column].to_numpy()
                active = _active_player_rows(frame)
                group_start = np.r_[True,
                    (replay[1:] != replay[:-1]) | (stints[1:] != stints[:-1])]
                group_id = np.cumsum(group_start, dtype=np.int64) - 1
                valid_rows = indices[
                    (next_slot[indices] >= 0) & (next_slot[indices] < len(PLAYER_SLOTS))
                    & np.isfinite(next_seconds[indices]) & (next_seconds[indices] > 0)
                ]
                selected_group_ids = group_id[valid_rows]

                for current_group in np.unique(selected_group_ids):
                    rows = np.flatnonzero(group_id == current_group)
                    group_rows = valid_rows[selected_group_ids == current_group]
                    targets = source_time[group_rows] + next_seconds[group_rows]
                    positions = np.searchsorted(source_time[rows], targets, side='left')
                    safe = np.minimum(positions, rows.size - 1)
                    previous = np.maximum(positions - 1, 0)
                    choose_previous = (positions >= rows.size) | (
                        np.abs(source_time[rows[previous]] - targets)
                        <= np.abs(source_time[rows[safe]] - targets)
                    )
                    nearest = np.where(choose_previous, previous, safe)
                    event_rows = rows[nearest]
                    slots = next_slot[group_rows]
                    invalid = (np.abs(source_time[event_rows] - targets) <= 0.5 / sample_rate)
                    invalid &= ~active[event_rows, slots]
                    next_slot[group_rows[invalid]] = -1
                    next_seconds[group_rows[invalid]] = np.nan
        else:
            contact = frame.select(tuple(f'player_{slot}_touching_ball' for slot in PLAYER_SLOTS))
            contact = contact.to_numpy().astype(np.float32, copy=False)
            contact = np.isfinite(contact) & (contact > 0.5)
            active = _active_player_rows(frame)
            contact &= active
            touching = np.where(contact.sum(axis=1) == 1, contact.argmax(axis=1), -1)
            source_time = frame['_source_time'].to_numpy().astype(np.float64, copy=False)
            stint_column = 'stint_id' if 'stint_id' in frame.columns else '_window_id'
            stints = frame[stint_column].to_numpy()
            replay_ids = frame['replay_id'].to_numpy()
            previous = np.r_[-1, touching[:-1]]
            boundary = np.r_[True,
                (stints[1:] != stints[:-1]) | (replay_ids[1:] != replay_ids[:-1])]
            starts = (touching >= 0) & ((touching != previous) | boundary)
            events = np.flatnonzero(starts)
            following = np.searchsorted(events, np.arange(frame.height), side='right')
            next_slot = np.full(frame.height, -1, dtype=np.int64)
            next_seconds = np.full(frame.height, np.nan, dtype=np.float32)
            rows = np.flatnonzero(following < events.size)
            future = events[following[rows]]
            same_stint = ((stints[rows] == stints[future])
                          & (replay_ids[rows] == replay_ids[future]))
            rows, future = rows[same_stint], future[same_stint]
            next_slot[rows] = touching[future]
            next_seconds[rows] = (source_time[future] - source_time[rows]).astype(np.float32)

        result['aux_next_touch_slot'] = torch.from_numpy(next_slot[indices])
        result['aux_next_touch_seconds'] = torch.from_numpy(next_seconds[indices])

    return result


class LGPSBallStateAux(nn.Module):
    @dataclass(frozen=True)
    class Settings:
        future_frames: tuple[int, ...] = (1, 5, 10, 20)
        hidden_dim: int | None = None
        num_hidden_layers: int = 1
        dropout: float = 0.0
        activation: Literal['gelu', 'relu', 'silu'] = 'gelu'
        loss_beta: float = 1.0
        continuous_loss_weight: float = 1.0
        rotation_loss_weight: float = 1.0

        @property
        def task_type(self):
            return LGPSBallStateAux

        def build(self, d_model):
            frames = tuple(self.future_frames)

            if not frames or any(not isinstance(frame, int) or frame <= 0 for frame in frames):
                raise ValueError('ball state auxiliary task requires positive future_frames')

            if tuple(sorted(set(frames))) != frames:
                raise ValueError('ball state future_frames must be unique and increasing')

            if self.num_hidden_layers < 1:
                raise ValueError('ball state auxiliary num_hidden_layers must be positive')

            if self.loss_beta <= 0:
                raise ValueError('ball state auxiliary loss_beta must be positive')

            if min(self.continuous_loss_weight, self.rotation_loss_weight) < 0:
                raise ValueError('ball state auxiliary loss weights must be non-negative')

            if self.continuous_loss_weight + self.rotation_loss_weight == 0:
                raise ValueError('at least one ball state loss weight must be positive')

            return LGPSBallStateAux(d_model, frames, self)

    def __init__(self, d_model, future_frames, settings=None):
        super().__init__()
        self.future_frames = tuple(() if future_frames is None else future_frames)
        self.settings = settings or self.Settings(future_frames=self.future_frames)
        self.feedback_projection = nn.Linear(len(BALL_FEATURES), d_model, bias=False)
        self.heads = nn.ModuleDict({
            str(frame): _mlp(
                d_model, len(BALL_FEATURES), self.settings.hidden_dim,
                self.settings.num_hidden_layers, self.settings.dropout,
                self.settings.activation,
            )
            for frame in self.future_frames
        })

    def forward(self, pooled):
        # Feed back model outputs only. Future targets are loss-only, so
        # later horizons cannot leak ground-truth future ball states.
        predictions = {}
        previous = None

        for frame in self.future_frames:
            context = pooled if previous is None else pooled + self.feedback_projection(previous)
            previous = self.heads[str(frame)](context)
            predictions[f'ball_{frame}'] = previous

        return predictions

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
                                        targets[:, index], LGPSScaler.BALL_SCALE,
                                        beta=self.settings.loss_beta,
                                        continuous_weight=self.settings.continuous_loss_weight,
                                        rotation_weight=self.settings.rotation_loss_weight)
            for index, frame in enumerate(self.future_frames)
        }


class LGPSNextTouchAux(nn.Module):
    @dataclass(frozen=True)
    class Settings:
        hidden_dim: int | None = None
        num_hidden_layers: int = 0
        dropout: float = 0.0
        activation: Literal['gelu', 'relu', 'silu'] = 'gelu'
        identity_loss_weight: float = 1.0
        timing_loss_weight: float = 1.0
        timing_beta: float = 1.0

        @property
        def task_type(self):
            return LGPSNextTouchAux

        def build(self, d_model):
            if self.num_hidden_layers < 0:
                raise ValueError('next touch num_hidden_layers must be non-negative')

            if self.hidden_dim is not None and self.hidden_dim < 1:
                raise ValueError('next touch hidden_dim must be positive')

            if not 0.0 <= self.dropout < 1.0:
                raise ValueError('next touch dropout must lie in [0, 1)')

            if self.identity_loss_weight < 0 or self.timing_loss_weight < 0:
                raise ValueError('next touch loss weights must be non-negative')

            if self.identity_loss_weight + self.timing_loss_weight == 0:
                raise ValueError('at least one next touch loss weight must be positive')

            if self.timing_beta <= 0:
                raise ValueError('next touch timing_beta must be positive')

            return LGPSNextTouchAux(d_model, self)

    def __init__(self, d_model, settings=None):
        super().__init__()
        self.settings = settings or self.Settings()

        if self.settings.num_hidden_layers:
            hidden_dim = self.settings.hidden_dim or d_model
            self.feature_encoder = _mlp(
                d_model * 2, hidden_dim, hidden_dim, self.settings.num_hidden_layers,
                self.settings.dropout, self.settings.activation,
            )
            self.player_score = nn.Linear(hidden_dim, 1)
            self.player_time = nn.Linear(hidden_dim, 1)
            self.global_score = None
        else:
            self.feature_encoder = None
            self.player_score = nn.Linear(d_model, 1)
            self.global_score = nn.Linear(d_model, d_model)
            self.player_time = nn.Linear(d_model, 1)

    def forward(self, player_streams, pooled):
        # A shared scorer over player tokens keeps touch identity equivariant
        # when players of the same team exchange input slots.
        if self.feature_encoder is not None:
            global_features = pooled[:, None].expand_as(player_streams)
            conditioned = self.feature_encoder(torch.cat((player_streams, global_features), dim=-1))
        else:
            conditioned = player_streams + self.global_score(pooled)[:, None]

        logits = self.player_score(conditioned).squeeze(-1)
        seconds = F.softplus(self.player_time(conditioned).squeeze(-1))
        return logits, seconds

    def loss(self, predictions, slot, seconds):
        logits, predicted_seconds = predictions
        valid = (slot >= 0) & (slot < logits.shape[-1]) & torch.isfinite(seconds) & (seconds > 0)
        count = valid.sum()
        global_count = count.detach().clone()
        world_size = 1

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        if bool(valid.any().item()):
            identity_sum = F.cross_entropy(logits[valid].float(), slot[valid], reduction='sum')
            selected_time = predicted_seconds[valid].gather(1, slot[valid, None]).squeeze(1)
            timing_sum = F.smooth_l1_loss(
                torch.log1p(selected_time.float()), torch.log1p(seconds[valid].float()),
                beta=self.settings.timing_beta, reduction='sum',
            )
        else:
            identity_sum = logits.sum() * 0.0
            timing_sum = predicted_seconds.sum() * 0.0

        identity = identity_sum * world_size / global_count.clamp_min(1)
        timing = timing_sum * world_size / global_count.clamp_min(1)
        return (self.settings.identity_loss_weight * identity
                + self.settings.timing_loss_weight * timing), bool(count.item())


class LGPSGradNorm:
    def __init__(self, task_names, alpha=1.5, momentum=0.9, device='cpu'):
        self.task_names = tuple(task_names)
        self.alpha = float(alpha)
        self.momentum = float(momentum)
        self.initial = torch.full((len(task_names) + 1,), float('nan'), device=device)
        self.weights = torch.ones(len(task_names), device=device)
        self.reported_weights = {name: 1.0 for name in self.task_names}

    def combine(self, primary, auxiliary, shared, *, update=True):
        if update:
            # Balance the auxiliary gradient budget against the primary task
            # using their gradients at the shared entity representation.
            losses = [primary] + [auxiliary[name][0] for name in self.task_names]
            active = torch.tensor([True] + [auxiliary[name][1] for name in self.task_names],
                                  device=shared.device, dtype=torch.int32)
            norms = []

            for loss in losses:
                gradient = torch.autograd.grad(
                    loss, shared, retain_graph=True, allow_unused=True,
                )[0]
                norms.append(torch.linalg.vector_norm(gradient.float()) if gradient is not None
                             else shared.new_zeros((), dtype=torch.float32))

            norms = torch.stack(norms).detach()
            values = torch.stack([loss.detach().float() for loss in losses])

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(norms.square_())
                dist.all_reduce(values)
                dist.all_reduce(active, op=dist.ReduceOp.MAX)
                norms = (norms / dist.get_world_size()).sqrt_()
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

        if update:
            self.reported_weights = {
                name: float(self.weights[index].item())
                for index, name in enumerate(self.task_names)
            }

        return combined, self.reported_weights


def configured_aux_tasks(tasks):
    configured = {}

    for settings in tasks or ():
        task_type = getattr(settings, 'task_type', None)

        if task_type not in (LGPSBallStateAux, LGPSNextTouchAux):
            raise TypeError(f'unsupported auxiliary task settings: {type(settings).__name__}')

        if task_type in configured:
            raise ValueError(f'{task_type.__name__} may only be configured once')

        configured[task_type] = settings

    return configured
