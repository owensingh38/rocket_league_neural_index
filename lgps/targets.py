import math
import numpy as np
import torch

DEFAULT_HORIZON_MAX = 30.0
OPEN_HORIZON_BUCKET_SECONDS = DEFAULT_HORIZON_MAX

def _horizon_steps(horizon_max):
    if not math.isfinite(horizon_max) or horizon_max < 1:
        raise ValueError("horizon_max must be finite and at least one second")
    return int(math.floor(horizon_max))


def open_horizon(horizon_seconds):
    return horizon_seconds is None or horizon_seconds == 0


def build_frame_batch(frame, indices, tokenizer, *, targets_per_team=1, geometric_width=1.0,
                      horizon_seconds=10.0,
                      horizon_max=DEFAULT_HORIZON_MAX,
                      two_way_prediction=False,
                      ):
    indices = np.asarray(indices, dtype=np.int64)
    goal_times = frame["time_until_next_goal"].to_numpy()[indices].astype(np.float64, copy=False)
    blue_goal_next = frame["blue_goal_next"].to_numpy()[indices].astype(bool, copy=False)
    orange_goal_next = frame["orange_goal_next"].to_numpy()[indices].astype(bool, copy=False)
    temporal, collapsed = temporal_targets(
        goal_times, blue_goal_next, orange_goal_next,
        targets_per_team=targets_per_team, geometric_width=geometric_width,
        horizon_seconds=horizon_seconds, horizon_max=horizon_max,
        two_way_prediction=two_way_prediction,
    )
    if two_way_prediction:
        # NumPy treats a CPU torch.bool tensor as integer indices via its
        # __array__ protocol (e.g. [True] becomes index [1]). Convert the
        # mask to a NumPy bool array before indexing NumPy inputs, then back
        # to a torch mask for the target tensors.
        valid = (collapsed >= 0).numpy()
        valid_tensor = torch.from_numpy(valid)
        indices = indices[valid]
        goal_times = goal_times[valid]
        blue_goal_next = blue_goal_next[valid]
        orange_goal_next = orange_goal_next[valid]
        temporal = temporal[valid_tensor]
        collapsed = collapsed[valid_tensor]
        if not indices.size:
            return None
    result = tokenizer(frame[indices.tolist()])
    result["temporal_targets"] = temporal
    result["collapsed_targets"] = collapsed
    if open_horizon(horizon_seconds):
        goal_teams = np.where(blue_goal_next, 0, np.where(orange_goal_next, 1, 2))
        horizons = np.arange(1, _horizon_steps(horizon_max) + 1, dtype=np.float64)[None, :]
        occurs_by_horizon = (
            np.isfinite(goal_times[:, None])
            & (goal_times[:, None] > 0.0)
            & (goal_times[:, None] <= horizons)
            & (blue_goal_next | orange_goal_next)[:, None]
        )
        horizon_labels = np.full((indices.size, horizons.shape[1]), 2, dtype=np.int64)
        horizon_labels[occurs_by_horizon] = np.broadcast_to(
            goal_teams[:, None], horizon_labels.shape
        )[occurs_by_horizon]
        result["horizon_targets"] = torch.from_numpy(horizon_labels)
    return result


def output_count(targets_per_team, horizon_seconds=10.0, two_way_prediction=False):
    if targets_per_team < 1:
        raise ValueError("targets_per_team must be at least one")
    if not open_horizon(horizon_seconds) and (not math.isfinite(horizon_seconds) or horizon_seconds < 0):
        raise ValueError("horizon_seconds must be non-negative and finite, or None")
    return 2 * int(targets_per_team) + int(not two_way_prediction)


def bucket_edges(targets_per_team=1, geometric_width=1.0, horizon_seconds=10.0,
                 horizon_max=DEFAULT_HORIZON_MAX):
    if (targets_per_team < 1 or not math.isfinite(geometric_width) or geometric_width <= 0
            or not math.isfinite(horizon_max) or horizon_max < 1
            or (not open_horizon(horizon_seconds)
                and (not math.isfinite(horizon_seconds) or horizon_seconds < 0))):
        raise ValueError("targets_per_team and horizon_max must be positive; geometric_width must be finite and positive")
    finite_buckets = targets_per_team - int(open_horizon(horizon_seconds))
    edges = np.empty(finite_buckets + 1, dtype=np.float64)
    edges[0] = 0.0
    if finite_buckets == 0:
        return edges
    raw = np.power(float(geometric_width), np.arange(finite_buckets, dtype=np.float64))
    duration = float(horizon_max) if open_horizon(horizon_seconds) else float(horizon_seconds)
    edges[1:] = np.cumsum(raw / raw.sum() * duration)
    edges[-1] = duration
    return edges


def temporal_targets(
    time_until_next_goal,
    blue_goal_next,
    orange_goal_next,
    *,
    targets_per_team=1,
    geometric_width=1.0,
    horizon_seconds=10.0,
    horizon_max=DEFAULT_HORIZON_MAX,
    sigma_base=1e-5,
    sigma_growth=0.15,
    two_way_prediction=False,
):
    dt = np.asarray(time_until_next_goal, dtype=np.float64)

    blue = np.asarray(blue_goal_next, dtype=bool)
    orange = np.asarray(orange_goal_next, dtype=bool)
    if dt.shape != blue.shape or dt.shape != orange.shape or np.any(blue & orange):
        raise ValueError("blue_goal_next and orange_goal_next must align and cannot both be true")
    if np.any((blue | orange) & ~np.isfinite(dt)):
        raise ValueError("a scoring stint must have a finite time_until_next_goal")

    teams = np.where(blue, 0, np.where(orange, 1, -1)).astype(np.int64)
    unbounded = open_horizon(horizon_seconds)
    edges = bucket_edges(targets_per_team, geometric_width, horizon_seconds, horizon_max)
    values = np.zeros((dt.size, output_count(
        targets_per_team, horizon_seconds, two_way_prediction,
    )), dtype=np.float32)
    if not two_way_prediction:
        values[:, -1] = 1.0
    qualifies_by_time = (np.isfinite(dt) & (dt > 0.0)
                         & ((dt <= horizon_seconds) if not unbounded else True))
    qualifying = qualifies_by_time & np.isin(teams, (0, 1))
    collapsed = np.where(qualifying, teams, -1 if two_way_prediction else 2).astype(np.int64)
    rows = np.flatnonzero(qualifying)

    if rows.size:
        mu = torch.from_numpy(dt[rows]).to(torch.float64)[:, None]
        sigma = float(sigma_base) + float(sigma_growth) * mu
        edge_tensor = torch.from_numpy(edges).to(torch.float64)[None, :]
        cdf = 0.5 * (1.0 + torch.erf((edge_tensor - mu) / (sigma * math.sqrt(2.0))))
        mass = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0).numpy().astype(np.float32)
        left_tail = cdf[:, 0].numpy().astype(np.float32)
        valid_values = np.zeros((rows.size, output_count(
            targets_per_team, horizon_seconds, two_way_prediction,
        )), dtype=np.float32)
        team_width = targets_per_team
        finite_buckets = targets_per_team - int(unbounded)

        for team in (0, 1):
            selected = teams[rows] == team
            start = team * team_width
            team_mass = mass[selected].copy()
            if unbounded and finite_buckets:
                team_mass[:, 0] += left_tail[selected]
            valid_values[selected, start:start + finite_buckets] = team_mass

        tail = (np.clip(1.0 - cdf[:, -1].numpy(), 0.0, 1.0).astype(np.float32)
                if unbounded
                else np.clip(1.0 - mass.sum(axis=1), 0.0, 1.0))
        if unbounded:
            if finite_buckets == 0:
                # One open-horizon bucket represents the whole scoring stint.
                tail.fill(1.0)
            long_bucket = targets_per_team - 1
            for team in (0, 1):
                selected = teams[rows] == team
                valid_values[selected, team * targets_per_team + long_bucket] = tail[selected]
        elif not two_way_prediction:
            valid_values[:, -1] = tail
        else:
            totals = valid_values.sum(axis=1, keepdims=True)
            valid_values = np.divide(
                valid_values, totals, out=np.zeros_like(valid_values),
                where=totals > 0.0,
            )

        values[rows] = valid_values

    targets = torch.from_numpy(values)

    expected_sums = torch.ones(dt.size)
    if two_way_prediction:
        expected_sums = torch.from_numpy(qualifying.astype(np.float32))
    if not torch.allclose(targets.sum(dim=-1), expected_sums, atol=1e-5):
        raise RuntimeError("temporal targets must sum to one")

    return targets, torch.from_numpy(collapsed)


def collapse_probabilities(probabilities, targets_per_team, horizon_seconds=10.0,
                           two_way_prediction=False):
    full_count = output_count(targets_per_team, horizon_seconds, two_way_prediction)
    if probabilities.shape[-1] != full_count:
        raise ValueError("probability width does not match targets_per_team")
    team_width = targets_per_team
    teams = (
        probabilities[..., :team_width].sum(dim=-1),
        probabilities[..., team_width:2 * team_width].sum(dim=-1),
    )
    if two_way_prediction:
        neither = probabilities.new_zeros(probabilities.shape[:-1])
    else:
        neither = probabilities[..., -1]
    return torch.stack((*teams, neither), dim=-1)
