import math

import numpy as np
import torch
from .aux_tasks import LGPSBallStateAux, LGPSNextTouchAux, LGPSPlayerStateAux, build_aux_targets


def build_frame_batch(frame, indices, tokenizer, *, targets_per_team=10, geometric_width=1.25,
                      future_frames=(10, 30, 60, 100), sample_rate=10.0, include_aux=True,
                      aux_tasks=(LGPSBallStateAux, LGPSPlayerStateAux, LGPSNextTouchAux)):
    indices = np.asarray(indices, dtype=np.int64)
    result = tokenizer(frame[indices.tolist()])
    replay_ids = frame["replay_id"].to_numpy()[indices]
    temporal = torch.zeros(indices.size, output_count(targets_per_team), dtype=torch.float32)
    collapsed = torch.empty(indices.size, dtype=torch.long)

    for replay_id in np.unique(replay_ids):
        local = np.flatnonzero(replay_ids == replay_id)
        temporal[local], collapsed[local] = temporal_targets(
            frame["time_until_next_goal"].to_numpy()[indices[local]],
            frame["blue_goal_in_next_ten_seconds"].to_numpy()[indices[local]],
            frame["orange_goal_in_next_ten_seconds"].to_numpy()[indices[local]],
            frame["neither_goal_in_next_ten_seconds"].to_numpy()[indices[local]],
            targets_per_team=targets_per_team, geometric_width=geometric_width)

    result["temporal_targets"] = temporal
    result["collapsed_targets"] = collapsed
    if include_aux:
        result.update(build_aux_targets(frame, indices, future_frames, aux_tasks, sample_rate))

    return result


def output_count(targets_per_team):
    if targets_per_team < 1:
        raise ValueError("targets_per_team must be at least one")
    return 2 * int(targets_per_team) + 1


def bucket_edges(targets_per_team=10, geometric_width=1.25):
    if targets_per_team < 1 or not math.isfinite(geometric_width) or geometric_width <= 0:
        raise ValueError("targets_per_team must be positive and geometric_width finite and positive")
    raw = np.power(float(geometric_width), np.arange(targets_per_team, dtype=np.float64))
    edges = np.empty(targets_per_team + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:] = np.cumsum(raw / raw.sum() * 10.0)
    edges[-1] = 10.0
    return edges


def temporal_targets(
    time_until_next_goal,
    blue_goal,
    orange_goal,
    neither_goal,
    *,
    targets_per_team=10,
    geometric_width=1.25,
    sigma_base=1e-5,
    sigma_growth=0.15
):
    dt = np.asarray(time_until_next_goal, dtype=np.float64)

    blue, orange, neither = (
        np.asarray(value, dtype=bool)
        for value in (blue_goal, orange_goal, neither_goal)
    )

    if not (dt.shape == blue.shape == orange.shape == neither.shape) or np.any(
        blue.astype(np.int8) + orange.astype(np.int8) + neither.astype(np.int8) != 1
    ):
        raise ValueError("goal state columns must be aligned and exactly one-hot")

    teams = np.where(blue, 0, np.where(orange, 1, -1)).astype(np.int64)
    edges = bucket_edges(targets_per_team, geometric_width)
    values = np.zeros((dt.size, output_count(targets_per_team)), dtype=np.float32)
    values[:, -1] = 1.0
    qualifying = np.isfinite(dt) & (dt > 0.0) & (dt <= 10.0) & np.isin(teams, (0, 1))
    collapsed = np.where(qualifying, teams, 2).astype(np.int64)
    rows = np.flatnonzero(qualifying)

    if rows.size:
        mu = torch.from_numpy(dt[rows]).to(torch.float64)[:, None]
        sigma = float(sigma_base) + float(sigma_growth) * mu
        edge_tensor = torch.from_numpy(edges).to(torch.float64)[None, :]
        cdf = 0.5 * (1.0 + torch.erf((edge_tensor - mu) / (sigma * math.sqrt(2.0))))
        mass = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0).numpy().astype(np.float32)
        valid_values = np.zeros((rows.size, output_count(targets_per_team)), dtype=np.float32)

        for team in (0, 1):
            selected = teams[rows] == team
            valid_values[selected, team * targets_per_team:(team + 1) * targets_per_team] = mass[selected]

        valid_values[:, -1] = np.clip(1.0 - mass.sum(axis=1), 0.0, 1.0)
        values[rows] = valid_values

    targets = torch.from_numpy(values)

    if not torch.allclose(targets.sum(dim=-1), torch.ones(dt.size), atol=1e-5):
        raise RuntimeError("temporal targets must sum to one")

    return targets, torch.from_numpy(collapsed)


def collapse_probabilities(probabilities, targets_per_team):
    if probabilities.shape[-1] != output_count(targets_per_team):
        raise ValueError("probability width does not match targets_per_team")
    return torch.stack(
        (
            probabilities[..., :targets_per_team].sum(dim=-1),
            probabilities[..., targets_per_team:2 * targets_per_team].sum(dim=-1),
            probabilities[..., -1],
        ),
        dim=-1,
    )
