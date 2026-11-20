from __future__ import annotations

import os
import pathlib
import re
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import polars as pl

from shared import BALL_FEATURES, BOOST_FEATURES, CONTEXT_FEATURES, GOAL_FEATURES, PLAYER_FEATURES
from . import inference as lgps_inference

TEAM_COLORS = {0: "#188cff", 1: "#ff8c18", 2: "#b7c4d1"}
TYPE_COLORS = {
    "ball": "#f4f7fb",
    "player": "#ffffff",
    "goal": "#e9eff7",
    "big_boost": "#ffd166",
    "small_boost": "#a6e3a1",
}

def _frame_source(frames: pl.DataFrame | str | Path, directory: Path) -> Path:
    if isinstance(frames, pl.DataFrame):
        if frames.is_empty():
            raise ValueError("frames must contain at least one replay frame")

        source = directory / "replay_frames.parquet"
        frames.write_parquet(source)
        return source

    source = Path(frames).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if source.suffix.lower() not in (".csv", ".parquet"):
        raise ValueError("frames path must point to a CSV or Parquet frame table")

    return source


def _load_model(model_path: str | Path, device: str, threads: int | None):
    import torch

    from .model import LGPSTransformer, load_lgps_checkpoint, load_lgps_weights
    from .config import LGPSModelConfig

    path = Path(model_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.name.endswith(".torchscript.pt") or path.suffix.lower() != ".pt":
        raise ValueError("Attention visualization requires a native LGPS .pt model")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")

    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be positive")

        torch.set_num_threads(threads)

    original_posix_path = pathlib.PosixPath

    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        checkpoint = load_lgps_checkpoint(path, map_location="cpu")
    finally:
        pathlib.PosixPath = original_posix_path

    config, state = LGPSModelConfig(**checkpoint["model_config"]), checkpoint["model_state_dict"]
    aux_tasks = checkpoint.get("train_config", {}).get("aux_tasks")
    model = LGPSTransformer(config, aux_tasks=aux_tasks).to(device).eval()
    load_lgps_weights(model, state, include_aux=bool(aux_tasks))

    for block in model.blocks:
        block.attention.capture_attention = True

    return model, torch


def _tokens(frame: pl.DataFrame, values: dict[str, np.ndarray], model,
            player_names: dict[int, str]) -> list[dict[str, Any]]:
    slots = lgps_inference._player_slots(frame)
    canonical: list[dict[str, Any]] = [
        {"index": 0, "label": "Ball", "kind": "ball", "team": 2,
         "xy": values["ball"][0, :2]},
    ]
    player_teams = values["entity_teams"][0, 1:7]

    for index, slot in enumerate(slots):
        if slot is None:
            canonical.append({"index": index + 1, "label": "Empty player slot",
                              "kind": "player", "team": 2, "xy": np.zeros(2)})
        else:
            team = int(player_teams[index])
            player_name = player_names.get(slot, "")
            canonical.append({
                "index": index + 1,
                "label": player_name or "Player",
                "kind": "player",
                "team": team,
                "xy": values["players"][0, index, :2],
            })

    for index, team in enumerate((0, 1)):
        canonical.append({
            "index": 7 + index,
            "label": ("Blue Goal", "Orange Goal")[index],
            "kind": "goal",
            "team": team,
            "xy": values["goals"][0, index, :2],
        })

    big_positions = (values["big_boosts"][0, :, :2] if "big_boosts" in values
                     else np.zeros((6, 2), dtype=np.float32))

    for index, position in enumerate(big_positions):
        canonical.append({"index": 9 + index, "label": f"Big Boost {index + 1}",
                          "kind": "big_boost", "team": 2, "xy": position})

    small_positions = (values["small_boosts"][0, :, :2] if "small_boosts" in values
                       else np.zeros((28, 2), dtype=np.float32))

    for index, position in enumerate(small_positions):
        canonical.append({"index": 15 + index, "label": f"Small Boost {index + 1}",
                          "kind": "small_boost", "team": 2, "xy": position})

    entity_indices = model.entity_indices.detach().cpu().tolist()
    return [canonical[index] for index in entity_indices]


def _player_name_map(frame: pl.DataFrame) -> dict[int, str]:
    rosters = {0: [], 1: []}

    for slot in range(1, 7):
        name_column = f"player_{slot}_name"
        team_column = f"player_{slot}_team_id"

        if name_column not in frame.columns or team_column not in frame.columns:
            continue

        names = frame[name_column].drop_nulls().unique().to_list()
        teams = frame[team_column].drop_nulls().unique().to_list()

        if len(names) != 1 or len(teams) != 1:
            continue

        team = int(teams[0])
        name = str(names[0]).strip()

        if team in rosters and name:
            rosters[team].append((slot, name))

    player_names = {}

    for team, roster in rosters.items():
        for player_index, (_, name) in enumerate(sorted(roster), start=1):
            player_names[team * 3 + player_index] = name

    return player_names


def _active_player_slots(frame: pl.DataFrame) -> set[int]:
    teams = {0: [], 1: []}

    for column in frame.columns:
        if not column.startswith("player_") or not column.endswith("_team_id"):
            continue

        values = frame[column].drop_nulls().unique().to_list()

        if len(values) == 1 and int(values[0]) in teams:
            teams[int(values[0])].append(int(column.split("_")[1]))

    return {
        team * 3 + index
        for team, roster in teams.items()
        for index, _ in enumerate(sorted(roster), start=1)
    }


def _attention_sequence(frame: pl.DataFrame, model, device: str, torch,
                       position_frame: pl.DataFrame | None = None,
                       player_names: dict[int, str] | None = None,
                       active_player_slots: set[int] | None = None):
    names = tuple(
        name for name in lgps_inference.INPUT_NAMES
        if name not in ("big_boosts", "small_boosts") or name in model.model_entities
    )
    values = lgps_inference._model_inputs(frame, names)
    tensors = {name: torch.from_numpy(value).to(device) for name, value in values.items()}
    head_attention_values = None
    head_attention_calls = []

    def make_attention_hook():
        def capture_attention(module, inputs, output):
            attention = module.last_attention
            head_attention_calls.append(
                attention.detach().to(device="cpu", dtype=torch.float16).numpy()
            )

        return capture_attention

    hooks = [
        block.attention.register_forward_hook(make_attention_hook())
        for block in model.blocks
    ]
    batch_size = 512

    try:
        for start in range(0, frame.height, batch_size):
            end = min(start + batch_size, frame.height)
            feed = {
                name: value if name == "entity_types" else value[start:end]
                for name, value in tensors.items() if name in names
            }
            head_attention_calls.clear()

            with torch.inference_mode():
                model(**feed)

            expected_calls = len(model.blocks) * model.loops_per_block

            if len(head_attention_calls) != expected_calls:
                raise RuntimeError(
                    f"Expected {expected_calls} per-head attention maps, captured {len(head_attention_calls)}"
                )

            batch_size_actual = end - start
            entity_count = len(model.entity_indices)

            if head_attention_values is None:
                head_attention_values = np.empty(
                    (frame.height, len(model.blocks), model.loops_per_block,
                     model.config.num_heads, entity_count, entity_count),
                    dtype=np.float16,
                )

            head_attention_values[start:end] = np.stack(
                head_attention_calls, axis=1,
            ).reshape(
                batch_size_actual, len(model.blocks), model.loops_per_block,
                model.config.num_heads, entity_count, entity_count,
            )
    finally:
        for hook in hooks:
            hook.remove()

    position_frame = frame if position_frame is None else position_frame
    position_values = (values if position_frame is frame
                       else lgps_inference._model_inputs(position_frame, names))
    row_values = {name: value[:1] for name, value in position_values.items()}
    token_templates = _tokens(
        position_frame.slice(0, 1), row_values, model, player_names or {},
    )
    canonical_positions = np.zeros((position_frame.height, 43, 3), dtype=np.float32)
    canonical_positions[:, 0] = position_values["ball"][:, :3]
    canonical_positions[:, 1:7] = position_values["players"][:, :, :3]
    canonical_positions[:, 7:9] = position_values["goals"][:, :, :3]

    if "big_boosts" in position_values:
        canonical_positions[:, 9:15] = position_values["big_boosts"][:, :, :3]

    if "small_boosts" in position_values:
        canonical_positions[:, 15:43] = position_values["small_boosts"][:, :, :3]

    positions = canonical_positions[:, model.entity_indices.detach().cpu().numpy()]
    keep_indices = [
        index for index, token in enumerate(token_templates)
        if token["kind"] != "player" or token["index"] in (active_player_slots or set())
    ]
    token_templates = [token_templates[index] for index in keep_indices]
    positions = positions[:, keep_indices]
    head_attention_values = head_attention_values[..., keep_indices, :][..., keep_indices]
    position_frame_numbers = (
        position_frame["frame_number"].cast(pl.Float64).to_numpy()
        if "frame_number" in position_frame.columns
        else np.arange(position_frame.height, dtype=np.float64)
    )
    return (
        head_attention_values, token_templates, positions,
        position_frame_numbers,
    )


class LGPSVisualizer(ABC):
    @abstractmethod
    def plot(self, frames, model_path, **kwargs):
        raise NotImplementedError

    def visualize(self, frames, model_path, **kwargs):
        return self.plot(frames, model_path, **kwargs)

    @staticmethod
    def heatmap(axis, matrix, title, x_label, y_label, x_ticks=None):
        from matplotlib.colors import TwoSlopeNorm

        matrix = np.asarray(matrix, dtype=np.float32)
        limit = max(float(np.nanmax(np.abs(matrix))), 1e-8)
        axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm",
                    norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
        axis.set_title(title, loc="left", fontsize=13, pad=10)
        axis.set_xlabel(x_label, fontsize=12, labelpad=8)
        axis.set_ylabel(y_label, fontsize=12, labelpad=8)
        axis.tick_params(axis="both", labelsize=10)

        if x_ticks is not None:
            axis.set_xticks(np.arange(len(x_ticks)), labels=x_ticks, rotation=90, fontsize=10)


class LGPSAttentionGraph(LGPSVisualizer):
    title = "LGPS Attention Network"

    def plot(self, 
        frames: pl.DataFrame | str | Path,
        model_path: str | Path = lgps_inference.MODEL_PATH,
        *,
        render_mode: str = "2d",
        sample_rate: float = 10.0,
        device: str = "cpu",
        threads: int | None = None,
        top_k: int = 1,
        playback_speed: float = 1.0,
        view_elev: float = 28,
        view_azim: float = -64,
        video_output_dir: str | Path | None = None,
        show: bool = True,
    ):
        render_mode = render_mode.lower()

        if render_mode not in ("2d", "3d"):
            raise ValueError("render_mode must be '2d' or '3d'")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not np.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError("playback_speed must be positive")

        from analyzerl_boxcars.animate import _clock_label

        model_path = Path(model_path).expanduser().resolve()
        original_posix_path = pathlib.PosixPath

        if os.name == "nt":
            # Map serialized POSIX paths while loading Linux-created checkpoints on Windows.
            pathlib.PosixPath = pathlib.WindowsPath

        try:
            with TemporaryDirectory(prefix="lgps_visualization_") as temporary:
                temporary_path = Path(temporary)
                source = _frame_source(frames, temporary_path)
                scored_paths = lgps_inference.apply_lgps(
                    source,
                    model_path=model_path,
                    output_dir=temporary_path / "scored",
                    device=device,
                    threads=threads,
                    sample_rate=sample_rate,
                )

                if not scored_paths:
                    raise ValueError("LGPS produced no estimates for the supplied frames")

                scored = lgps_inference._read_frame(scored_paths[0])
                source_frame = lgps_inference._read_frame(source)
                replay_id = (str(source_frame.get_column("replay_id")[0])
                             if "replay_id" in source_frame.columns else source.stem)
                source_columns = set(source_frame.columns)
                team_columns = [name for name in ("blue_team_name", "orange_team_name")
                                if name in source_columns]
                player_name_columns = [f"player_{slot}_name" for slot in range(1, 7)
                                       if f"player_{slot}_name" in source_columns]
                player_names = _player_name_map(source_frame)
                active_player_slots = _active_player_slots(source_frame)
                clock_column = next((name for name in
                                     ("frame_seconds_elapsed", "seconds_elapsed")
                                     if name in source_columns), None)
                blue_score_column = next((name for name in
                                          ("blue_team_score", "blue_score")
                                          if name in source_columns), None)
                orange_score_column = next((name for name in
                                            ("orange_team_score", "orange_score")
                                            if name in source_columns), None)
                active_column = "frame_is_active" if "frame_is_active" in source_columns else None
                metadata_columns = [name for name in (
                    "frame_number", clock_column, blue_score_column, orange_score_column,
                    active_column, *team_columns, *player_name_columns,
                ) if name is not None]
                estimates = scored.select(lgps_inference.PROBABILITY_COLUMNS).to_numpy()
                prediction_frame_numbers = (
                    scored["frame_number"].cast(pl.Float64).to_numpy()
                    if "frame_number" in scored.columns
                    else np.arange(scored.height, dtype=np.float64) * (30.0 / sample_rate)
                )
                source_metadata = (
                    source_frame.select(metadata_columns)
                    if metadata_columns else pl.DataFrame({"frame_number": prediction_frame_numbers})
                )
                position_frame = lgps_inference._prepare_if_needed(
                    source_frame, sample_rate=30.0,
                )
                del source_frame
                model, torch = _load_model(model_path, device, threads)

                (
                    head_attention_values, token_templates,
                    positions_by_frame, position_frame_numbers,
                ) = _attention_sequence(
                    scored, model, device, torch, position_frame=position_frame,
                    player_names=player_names,
                    active_player_slots=active_player_slots,
                )
                del position_frame
                team_row = source_metadata.select(team_columns).row(0) if team_columns else ()
                team_metadata = dict(zip(team_columns, team_row))
                source_frame_numbers = (
                    source_metadata["frame_number"].cast(pl.Float64).to_numpy()
                    if "frame_number" in source_metadata.columns else prediction_frame_numbers
                )
                source_elapsed = (
                    source_metadata[clock_column].cast(pl.Float64).fill_null(0.0).to_numpy()
                    if clock_column else source_frame_numbers / 30.0
                )
                source_scores = np.column_stack([
                    source_metadata[column].cast(pl.Float64).fill_null(0.0).to_numpy()
                    if column else np.zeros(source_metadata.height, dtype=np.float64)
                    for column in (blue_score_column, orange_score_column)
                ])
                source_active = (
                    source_metadata[active_column].cast(pl.Boolean).fill_null(False).to_numpy()
                    if active_column else np.ones(source_metadata.height, dtype=bool)
                )
        finally:
            pathlib.PosixPath = original_posix_path

        if len(estimates) != len(head_attention_values):
            raise RuntimeError("LGPS estimates and attention frames have different lengths")

        order = np.argsort(prediction_frame_numbers, kind="stable")
        prediction_frame_numbers = np.asarray(prediction_frame_numbers)[order]
        estimates = estimates[order]
        head_attention_values = head_attention_values[order]
        unique_frames, unique_indices = np.unique(prediction_frame_numbers, return_index=True)
        estimates = estimates[unique_indices]
        head_attention_values = head_attention_values[unique_indices]
        position_order = np.argsort(position_frame_numbers, kind="stable")
        position_frame_numbers = np.asarray(position_frame_numbers)[position_order]
        positions_by_frame = positions_by_frame[position_order]
        position_unique_frames, position_unique_indices = np.unique(
            position_frame_numbers, return_index=True,
        )
        position_frame_numbers = position_unique_frames
        positions_by_frame = positions_by_frame[position_unique_indices]
        source_order = np.argsort(source_frame_numbers, kind="stable")
        source_frame_numbers = np.asarray(source_frame_numbers)[source_order]
        source_elapsed = source_elapsed[source_order]
        source_scores = source_scores[source_order]
        source_active = source_active[source_order]
        source_unique_frames, source_unique_indices = np.unique(
            source_frame_numbers, return_index=True,
        )
        source_frame_numbers = source_unique_frames
        source_elapsed = source_elapsed[source_unique_indices]
        source_scores = source_scores[source_unique_indices]
        source_active = source_active[source_unique_indices]
        render_frames = source_frame_numbers.copy()

        def interpolate(values: np.ndarray, sample_frames: np.ndarray) -> np.ndarray:
            values = np.asarray(values)
            flat = values.reshape(values.shape[0], -1)
            interpolated = np.column_stack([
                np.interp(render_frames, sample_frames, flat[:, column])
                for column in range(flat.shape[1])
            ])
            return interpolated.reshape((len(render_frames),) + values.shape[1:])

        def interpolate_active_lgps(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values)
            flat = values.reshape(values.shape[0], -1)
            held_indices = np.clip(
                np.searchsorted(unique_frames, render_frames, side="right") - 1,
                0, len(unique_frames) - 1,
            )
            result = flat[held_indices].copy()
            boundaries = np.diff(np.r_[False, source_active, False].astype(np.int8))
            starts = np.flatnonzero(boundaries == 1)
            ends = np.flatnonzero(boundaries == -1)

            for start, end in zip(starts, ends):
                segment_frames = render_frames[start:end]
                sample_mask = (unique_frames >= segment_frames[0]) & (unique_frames <= segment_frames[-1])
                segment_samples = unique_frames[sample_mask]
                segment_values = flat[sample_mask]

                if not len(segment_samples):
                    nearest = int(np.argmin(np.abs(unique_frames - segment_frames[0])))
                    result[start:end] = flat[nearest]
                    continue

                for column in range(flat.shape[1]):
                    result[start:end, column] = np.interp(
                        segment_frames, segment_samples, segment_values[:, column],
                    )

            return result.reshape((len(render_frames),) + values.shape[1:])

        estimates = interpolate_active_lgps(estimates)
        position_indices = np.clip(
            np.searchsorted(position_frame_numbers, render_frames, side="right") - 1,
            0, len(position_frame_numbers) - 1,
        )
        positions_by_frame = positions_by_frame[position_indices]
        source_indices = np.clip(
            np.searchsorted(source_frame_numbers, render_frames, side="right") - 1,
            0, len(source_frame_numbers) - 1,
        )
        display_scores = source_scores[source_indices]
        elapsed_values = source_elapsed[source_indices]
        blue_team_name = str(team_metadata.get("blue_team_name") or "Blue")
        orange_team_name = str(team_metadata.get("orange_team_name") or "Orange")

        from analyzerl_boxcars.animate import BACK_WALL_Y, GOAL_HEIGHT

        for index, token in enumerate(token_templates):
            if token["kind"] == "goal":
                goal_y = -BACK_WALL_Y if token["team"] == 0 else BACK_WALL_Y
                positions_by_frame[:, index] = (0.0, goal_y, GOAL_HEIGHT / 2)

        animation = self._render_attention_matplotlib(
            render_mode, playback_speed, top_k, view_elev, view_azim,
            show, head_attention_values, unique_frames,
            source_active, token_templates, positions_by_frame, render_frames,
            estimates, display_scores, elapsed_values,
            blue_team_name, orange_team_name,
        )

        if video_output_dir is not None:
            from matplotlib.animation import FFMpegWriter

            output_dir = Path(video_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_replay_id = re.sub(r"[^\w.-]+", "_", replay_id).strip("._") or "replay"
            start_frame, end_frame = int(render_frames[0]), int(render_frames[-1])
            output_path = output_dir / (
                f"lgps_attention_graph_{safe_replay_id}_{start_frame}_{end_frame}.mp4"
            )
            animation.save(
                output_path,
                writer=FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]),
                dpi=90,
            )
            animation.output_path = output_path

        return animation




    def _render_attention_matplotlib(self, 
        render_mode, playback_speed, top_k, view_elev, view_azim,
        show, head_attention_values, prediction_frames,
        source_active, token_templates, positions_by_frame, render_frames,
        estimates, display_scores, elapsed_values,
        blue_team_name, orange_team_name,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.cm import ScalarMappable
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize, to_rgb
        from mpl_toolkits.mplot3d import proj3d
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from analyzerl_boxcars.animate import (
            _clock_label, _draw_boost_pads_2d, _draw_boost_pads_3d,
            _draw_field, _draw_field_3d,
        )

        is_3d = render_mode == "3d"
        figure = plt.figure(figsize=(18, 13), dpi=90)
        axis = figure.add_axes(
            (0.005, 0.115, 0.56, 0.84), projection="3d" if is_3d else None,
        )
        figure.patch.set_facecolor("#101722")
        axis.set_facecolor("#101722")

        if is_3d:
            _draw_field_3d(axis)
            _draw_boost_pads_3d(axis)
            axis.set_xlim(4096, -4096)
            axis.set_ylim(-6000, 6000)
            axis.set_zlim(0, 2048)
            axis.view_init(elev=view_elev, azim=view_azim)
            axis.set_box_aspect((8192, 12000, 3000))
            axis.grid(False)
        else:
            _draw_field(axis)
            _draw_boost_pads_2d(axis)
            axis.set_xlim(4096, -4096)
            axis.set_ylim(-6000, 6000)
            axis.set_aspect("equal", adjustable="box")

        axis.set_axis_off()
        color_map = plt.get_cmap("plasma")
        scale_axis = figure.add_axes((0.105, 0.062, 0.40, 0.014))
        scale = figure.colorbar(
            ScalarMappable(norm=Normalize(0.0, 1.0), cmap=color_map),
            cax=scale_axis, orientation="horizontal",
            ticks=(0.0, 0.5, 1.0),
        )
        scale.set_label("Relative Attention Weight (Replay-Wide Scale)", color="#d7e0ea", labelpad=3)
        scale.ax.tick_params(colors="#d7e0ea", labelsize=8)
        scale.outline.set_edgecolor("#718096")
        figure.suptitle("LGPS Attention Network", color="#f3f6fa", y=0.975, fontsize=14)

        block_count, loop_count, head_count = head_attention_values.shape[1:4]
        entity_count = head_attention_values.shape[-1]
        global_attention_scale = max(float(head_attention_values.max()), 1e-12)
        head_choices = {
            f"Block {block + 1} · Loop {loop + 1} · Head {head + 1}": (block, loop, head)
            for block in range(block_count)
            for loop in range(loop_count)
            for head in range(head_count)
        }
        selected_head = [(min(3, block_count - 1), min(1, loop_count - 1), 0)]
        current_frame = [0]

        def interpolate_head(block, loop, head):
            sampled_maps = np.asarray(
                head_attention_values[:, block, loop, head], dtype=np.float32,
            )
            flattened = sampled_maps.reshape(len(prediction_frames), -1)
            held = np.clip(
                np.searchsorted(prediction_frames, render_frames, side="right") - 1,
                0, len(prediction_frames) - 1,
            )
            result = flattened[held].copy()
            boundaries = np.diff(np.r_[False, source_active, False].astype(np.int8))
            starts = np.flatnonzero(boundaries == 1)
            ends = np.flatnonzero(boundaries == -1)

            for start, end in zip(starts, ends):
                frame_slice = render_frames[start:end]
                sample_mask = (prediction_frames >= frame_slice[0]) & (prediction_frames <= frame_slice[-1])
                sample_frames = prediction_frames[sample_mask]
                sample_values = flattened[sample_mask]

                if len(sample_frames) == 0:
                    nearest = int(np.argmin(np.abs(prediction_frames - frame_slice[0])))
                    result[start:end] = flattened[nearest]
                else:
                    for column in range(flattened.shape[1]):
                        result[start:end, column] = np.interp(
                            frame_slice, sample_frames, sample_values[:, column],
                        )

            return result.reshape(len(render_frames), entity_count, entity_count)

        animated_head_attention = np.empty(
            (len(render_frames), block_count, loop_count, head_count,
             entity_count, entity_count),
            dtype=np.float32,
        )

        for block in range(block_count):
            for loop in range(loop_count):
                for head in range(head_count):
                    animated_head_attention[:, block, loop, head] = interpolate_head(
                        block, loop, head,
                    )

        selected_maps = [interpolate_head(*selected_head[0])]
        attention_scale = [global_attention_scale]
        preview_axes = []
        preview_images = []
        preview_choices = {}
        preview_head_indices = []
        preview_rows = block_count * loop_count
        preview_columns = head_count
        preview_vmax = global_attention_scale
        preview_left, preview_bottom = 0.585, 0.01
        preview_width, preview_height = 0.39, 0.54
        preview_gap_x, preview_gap_y = 0.006, 0.006
        figure.text(
            preview_left + preview_width / 2, preview_bottom + preview_height + 0.012,
            "Head Number", color="#d7e0ea", fontsize=7, ha="center", va="bottom",
        )
        figure.text(
            preview_left - 0.014, preview_bottom + preview_height / 2,
            "Block / Loop", color="#d7e0ea", fontsize=7, ha="center", va="center",
            rotation=90,
        )
        cell_width = (preview_width - preview_gap_x * (preview_columns - 1)) / preview_columns
        cell_height = (preview_height - preview_gap_y * (preview_rows - 1)) / preview_rows

        for block in range(block_count):
            for loop in range(loop_count):
                row = block * loop_count + loop

                for head in range(head_count):
                    left = preview_left + head * (cell_width + preview_gap_x)
                    bottom = preview_bottom + (preview_rows - row - 1) * (cell_height + preview_gap_y)
                    preview_axis = figure.add_axes((left, bottom, cell_width, cell_height))
                    preview_axis.set_facecolor("#101722")
                    preview_image = preview_axis.imshow(
                        animated_head_attention[0, block, loop, head], cmap=color_map,
                        vmin=0, vmax=preview_vmax, interpolation="nearest", aspect="equal",
                    )

                    if row == 0:
                        preview_axis.set_title(
                            f"{head + 1}", color="#d7e0ea", fontsize=6, pad=1,
                        )

                    if head == 0:
                        preview_axis.set_ylabel(
                            f"B{block + 1} L{loop + 1}", color="#d7e0ea",
                            fontsize=5, rotation=0, labelpad=11, va="center",
                        )

                    preview_axis.set_xticks([])
                    preview_axis.set_yticks([])
                    choice = f"Block {block + 1} · Loop {loop + 1} · Head {head + 1}"
                    preview_choices[preview_axis] = choice
                    preview_head_indices.append((block, loop, head))

                    for spine in preview_axis.spines.values():
                        spine.set_color("#e6c85c" if (block, loop, head) == selected_head[0] else "#455365")

                    preview_axes.append(preview_axis)
                    preview_images.append(preview_image)

        main_head_axis = figure.add_axes((0.635, 0.63, 0.33, 0.32))
        main_head_axis.set_facecolor("#101722")
        main_head_image = main_head_axis.imshow(
            selected_maps[0][0], cmap=color_map, vmin=0,
            vmax=attention_scale[0], interpolation="nearest", aspect="equal",
        )
        token_names = [token["label"] for token in token_templates]
        token_indices = np.arange(entity_count)
        main_head_axis.set_xticks(
            token_indices, labels=[f"K: {name}" for name in token_names],
            rotation=90, fontsize=3, ha="center",
        )
        main_head_axis.set_yticks(
            token_indices, labels=[f"Q: {name}" for name in token_names], fontsize=3,
        )
        main_head_axis.set_xlabel("Key Token", color="#d7e0ea", fontsize=6, labelpad=0)
        main_head_axis.set_ylabel("Query Token", color="#d7e0ea", fontsize=6, labelpad=1)
        main_head_axis.tick_params(colors="#d7e0ea", length=1, pad=1)
        main_head_colorbar_axis = figure.add_axes((0.975, 0.63, 0.01, 0.32))
        main_head_colorbar = figure.colorbar(main_head_image, cax=main_head_colorbar_axis)
        main_head_colorbar.set_label("Head Attention Weight", color="#d7e0ea", fontsize=8)
        main_head_colorbar.ax.tick_params(colors="#d7e0ea", labelsize=6)
        initial_choice = next(
            choice for choice, indices in head_choices.items() if indices == selected_head[0]
        )
        main_head_axis.set_title(initial_choice, color="#f3f6fa", fontsize=8, pad=2)

        initial_positions = positions_by_frame[0]
        token_colors = [
            to_rgb(TEAM_COLORS[token["team"]]) if token["kind"] in ("player", "goal")
            else to_rgb(TYPE_COLORS[token["kind"]])
            for token in token_templates
        ]
        token_sizes = [
            {"ball": 70, "player": 80, "goal": 0, "big_boost": 35, "small_boost": 18}[token["kind"]]
            for token in token_templates
        ]

        if is_3d:
            node_artist = axis.scatter(
                initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2],
                s=token_sizes, c=token_colors, edgecolors="#101722", linewidths=0.7,
                depthshade=False, zorder=4,
            )
        else:
            node_artist = axis.scatter(
                initial_positions[:, 0], initial_positions[:, 1],
                s=token_sizes, c=token_colors, edgecolors="#101722", linewidths=0.7, zorder=4,
            )

        label_indices = [
            index for index, token in enumerate(token_templates) if token["kind"] == "player"
        ]
        labels = []

        for index in label_indices:
            if is_3d:
                label = axis.annotate(
                    token_templates[index]["label"], xy=(0, 0), xytext=(0, 6),
                    xycoords=axis.transData, textcoords="offset points",
                    fontsize=7, color="#f3f6fa", ha="center", va="bottom",
                    annotation_clip=False,
                )
            else:
                label = axis.text(
                    initial_positions[index, 0], initial_positions[index, 1] + 85,
                    token_templates[index]["label"], fontsize=7, color="#f3f6fa",
                    ha="center", va="bottom", rotation=0,
                )

            labels.append(label)

        initial_segments = np.asarray(
            [[initial_positions[0], initial_positions[0]]], dtype=np.float32,
        )

        if is_3d:
            edges = Line3DCollection(
                initial_segments, colors=["#101722"], linewidths=[0.0], zorder=2,
            )
            axis.add_collection3d(edges)
        else:
            edges = LineCollection(initial_segments[:, :, :2], colors=["#101722"], linewidths=[0.0], zorder=2)
            axis.add_collection(edges)

        status_text = axis.text2D if is_3d else axis.text
        status = status_text(
            0.02, 0.97, "", transform=axis.transAxes, ha="left", va="top",
            color="#f3f6fa", fontsize=10,
        )
        zoom_scale = 1.0
        scalable_collections = [
            (collection, np.asarray(collection.get_sizes(), dtype=np.float64).copy())
            for collection in axis.collections if hasattr(collection, "get_sizes")
        ]

        def zoom_field(event):
            nonlocal zoom_scale

            if event.inaxes is not axis:
                return

            scale = 0.82 if event.button == "up" else 1.22
            previous_zoom = zoom_scale
            zoom_scale *= 1.0 / scale

            for get_limits, set_limits in (
                (axis.get_xlim3d if is_3d else axis.get_xlim, axis.set_xlim3d if is_3d else axis.set_xlim),
                (axis.get_ylim3d if is_3d else axis.get_ylim, axis.set_ylim3d if is_3d else axis.set_ylim),
            ):
                lower, upper = get_limits()
                center = (lower + upper) / 2
                half_range = (upper - lower) * scale / 2
                set_limits(center - half_range, center + half_range)

            for collection, base_sizes in scalable_collections:
                collection.set_sizes(base_sizes * zoom_scale)

            if edges.get_linewidths().size:
                edges.set_linewidths(edges.get_linewidths() * zoom_scale / previous_zoom)

            for label in labels:
                label.set_fontsize(7 * np.sqrt(zoom_scale))

            if is_3d:
                lower, upper = axis.get_zlim3d()
                center = (lower + upper) / 2
                half_range = (upper - lower) * scale / 2
                axis.set_zlim3d(center - half_range, center + half_range)

            figure.canvas.draw_idle()

        figure.canvas.mpl_connect("scroll_event", zoom_field)

        def draw(frame_index):
            current_frame[0] = frame_index
            positions = positions_by_frame[frame_index]
            attention = selected_maps[0][frame_index]
            main_head_image.set_data(attention)

            for preview_image, (block, loop, head) in zip(preview_images, preview_head_indices):
                preview_image.set_data(animated_head_attention[frame_index, block, loop, head])

            edge_sources = []
            edge_targets = []

            for query_index, row_values in enumerate(attention):
                row_values = row_values.copy()
                row_values[query_index] = -np.inf
                count = min(top_k, len(row_values) - 1)
                candidates = np.argpartition(row_values, -count)[-count:]
                candidates = sorted(candidates, key=row_values.__getitem__, reverse=True)
                edge_sources.extend(int(key) for key in candidates)
                edge_targets.extend([query_index] * len(candidates))

            if edge_sources:
                source_array = np.asarray(edge_sources, dtype=np.int64)
                target_array = np.asarray(edge_targets, dtype=np.int64)
                segments = np.stack((positions[source_array], positions[target_array]), axis=1)
                weights = attention[target_array, source_array]
                relative_weights = np.clip(weights / attention_scale[0], 0.0, 1.0)
                edges.set_segments(segments if is_3d else segments[:, :, :2])
                edges.set_colors(color_map(relative_weights))
                edges.set_linewidths((0.45 + 0.85 * relative_weights) * zoom_scale)

            if is_3d:
                node_artist._offsets3d = (
                    positions[:, 0], positions[:, 1], positions[:, 2],
                )
                projection = axis.get_proj()

                for index, label in zip(label_indices, labels):
                    projected_x, projected_y, _ = proj3d.proj_transform(
                        positions[index, 0], positions[index, 1], positions[index, 2],
                        projection,
                    )
                    label.xy = (projected_x, projected_y)
            else:
                node_artist.set_offsets(positions[:, :2])

                for index, label in zip(label_indices, labels):
                    label.set_position((positions[index, 0], positions[index, 1] + 85))

            blue_probability, orange_probability, neither_probability = estimates[frame_index]
            blue_score, orange_score = display_scores[frame_index]
            status.set_text(
                f"{blue_team_name} {blue_score:.0f} – {orange_score:.0f} {orange_team_name}"
                f"  |  {_clock_label(float(elapsed_values[frame_index]))}"
                f"  |  Frame {render_frames[frame_index]:.0f}  "
                f"Blue {blue_probability:.1%}  Orange {orange_probability:.1%}  Neither {neither_probability:.1%}"
            )
            return (edges, node_artist, *labels, status)

        def update_head(choice):
            block, loop, head = head_choices[choice]
            selected_head[0] = (block, loop, head)
            selected_maps[0] = animated_head_attention[:, block, loop, head]
            attention_scale[0] = global_attention_scale
            main_head_axis.set_title(choice, color="#f3f6fa", fontsize=8, pad=2)
            main_head_image.set_clim(0, attention_scale[0])
            main_head_colorbar.update_normal(main_head_image)

            for preview_axis, preview_image in zip(preview_axes, preview_images):
                preview_choice = preview_choices[preview_axis]
                preview_image.set_data(animated_head_attention[
                    (current_frame[0], *head_choices[preview_choice])
                ])

                for spine in preview_axis.spines.values():
                    spine.set_color("#e6c85c" if head_choices[preview_choice] == (block, loop, head) else "#455365")

            main_head_image.set_data(selected_maps[0][current_frame[0]])
            draw(current_frame[0])
            figure.canvas.draw_idle()

        animation = FuncAnimation(
            figure, draw, frames=len(render_frames), interval=1000.0 / 30.0,
            init_func=lambda: draw(0), blit=False, repeat=False, cache_frame_data=False,
        )

        if show:
            manager = plt.get_current_fig_manager()

            if hasattr(manager, "window") and hasattr(manager.window, "state"):
                manager.window.state("zoomed")

            try:
                from tkinter import ttk

                window = manager.window
                canvas_widget = figure.canvas.get_tk_widget()
                selector_frame = ttk.Frame(window)
                selector_frame.pack(side="top", fill="x", before=canvas_widget)
                ttk.Label(selector_frame, text="Attention Head").pack(side="left", padx=(10, 5), pady=3)
                head_selector = ttk.Combobox(
                    selector_frame, values=tuple(head_choices), state="readonly", width=32,
                )
                head_selector.pack(side="left", padx=5, pady=3)
                head_selector.current(tuple(head_choices).index(initial_choice))
                head_selector.bind("<<ComboboxSelected>>", lambda event: update_head(head_selector.get()))
                update_head(initial_choice)
            except (ImportError, AttributeError, TypeError):
                pass

            plt.show(block=True)

        return animation



class LGPSInputFeatureHeatmaps(LGPSVisualizer):
    title = "LGPS Input Feature Weights"
    encoder_titles = {
        "ball": "Ball", "player": "Player", "goal": "Goal",
        "big_boost": "Big Boost", "small_boost": "Small Boost", "context": "Context",
    }

    def plot(self, frames, model_path, *, device="cpu", threads=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        model, _ = _load_model(model_path, device, threads)
        encoders = _encoder_weights(model)

        if not encoders:
            raise ValueError("The selected model has no input encoders to visualize")

        figure, axes = plt.subplots(len(encoders), 1, figsize=(22, 4.2 * len(encoders)), squeeze=False)
        figure.suptitle(self.title, y=0.995)

        for row, (name, feature_names, weights) in enumerate(encoders):
            axes[row, 0].imshow(weights, aspect="auto", interpolation="nearest", cmap="coolwarm")
            axes[row, 0].set_title(f"{self.encoder_titles[name]} Encoder Weights")
            axes[row, 0].set_xlabel("Input Feature Index")
            axes[row, 0].set_ylabel("Hidden Neuron")
            axes[row, 0].set_xticks(np.arange(len(feature_names)), labels=np.arange(len(feature_names)), fontsize=7)

        figure.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.04, hspace=0.32)

        if show:
            plt.show()

        return figure


class LGPSWeightMagnitudes(LGPSVisualizer):
    title = "LGPS Feature Weight Magnitudes"
    encoder_titles = LGPSInputFeatureHeatmaps.encoder_titles

    def plot(self, frames, model_path, *, device="cpu", threads=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        model, _ = _load_model(model_path, device, threads)
        encoders = _encoder_weights(model)

        if not encoders:
            raise ValueError("The selected model has no input encoders to visualize")

        figure, axes = plt.subplots(len(encoders), 1, figsize=(22, 4.2 * len(encoders)), squeeze=False)
        figure.suptitle(self.title, y=0.995)

        for axis, (name, feature_names, weights) in zip(axes[:, 0], encoders):
            axis.bar(np.arange(len(feature_names)), np.linalg.norm(weights, axis=0))
            axis.set_title(f"{self.encoder_titles[name]} ({len(feature_names)} Inputs)", loc="left")
            axis.set_xlabel("Input Feature Index")
            axis.set_ylabel("L2 Norm")
            axis.set_xticks(np.arange(len(feature_names)), labels=np.arange(len(feature_names)), fontsize=7)

        figure.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.04, hspace=0.32)

        if show:
            plt.show()

        return figure


class LGPSGeometryBiasWeights(LGPSVisualizer):
    title = "LGPS Geometry Bias MLP Weights"

    def plot(self, frames, model_path, *, device="cpu", threads=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        model, _ = _load_model(model_path, device, threads)

        if model.config.attention_mode != "relational":
            raise ValueError("Geometry bias weights require relational attention")

        figure, axes = plt.subplots(len(model.blocks), 4, figsize=(36, 6 * len(model.blocks)), squeeze=False)
        figure.suptitle(self.title, y=0.995)

        for row, block in enumerate(model.blocks):
            attention = block.attention
            self.heatmap(axes[row, 0], attention.geometry_bias[0].weight.detach().float().cpu().numpy(),
                         f"Block {row + 1}: Geometry Input Layer", "Geometry Feature Index", "Hidden Unit")
            last = attention.geometry_bias[-1].weight.detach().float().cpu().numpy()
            self.heatmap(axes[row, 1], last.T, f"Block {row + 1}: Geometry Output Layer",
                         "Attention Head", "Hidden Unit", [f"Head {i + 1}" for i in range(last.shape[0])])
            pad = attention.pad_geometry_bias

            if pad is None:
                for axis in axes[row, 2:]:
                    axis.set_visible(False)
            else:
                self.heatmap(axes[row, 2], pad[0].weight.detach().float().cpu().numpy(),
                             f"Block {row + 1}: Boost-Pad Input Layer", "Pad Geometry Feature Index", "Hidden Unit")
                pad_last = pad[-1].weight.detach().float().cpu().numpy()
                self.heatmap(axes[row, 3], pad_last.T, f"Block {row + 1}: Boost-Pad Output Layer",
                             "Attention Head", "Hidden Unit", [f"Head {i + 1}" for i in range(pad_last.shape[0])])

        figure.subplots_adjust(left=0.055, right=0.99, top=0.97, bottom=0.08, hspace=0.5, wspace=0.32)

        if show:
            plt.show()

        return figure


class LGPSPairwiseStateWeights(LGPSVisualizer):
    title = "LGPS Pairwise-State Bias Weights"
    role_names = ["Ball Type", "Player Type", "Goal Type", "Boost Pad Type", "Blue Team", "Orange Team",
                  "Neutral Team", "Ball Role", "Blue Player Role", "Orange Player Role", "Blue Goal Role",
                  "Orange Goal Role", "Neutral Entity Role"]

    def plot(self, frames, model_path, *, device="cpu", threads=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        model, _ = _load_model(model_path, device, threads)

        if model.config.attention_mode != "relational":
            raise ValueError("Pairwise-state weights require relational attention")

        figure, axes = plt.subplots(len(model.blocks), 5, figsize=(40, 6.2 * len(model.blocks)), squeeze=False)
        figure.suptitle(self.title, y=0.995)

        for row, block in enumerate(model.blocks):
            attention = block.attention
            heads, rank = attention.num_heads, attention.pairwise_rank
            source = attention.pairwise_source.weight.detach().float().cpu().numpy().reshape(heads, rank, -1)
            target = attention.pairwise_target.weight.detach().float().cpu().numpy().reshape(heads, rank, -1)
            self.heatmap(axes[row, 0], source.reshape(heads * rank, -1), f"Block {row + 1}: Source Projection",
                         "Model Dimension", "Head × Pairwise Rank")
            self.heatmap(axes[row, 1], target.reshape(heads * rank, -1), f"Block {row + 1}: Target Projection",
                         "Model Dimension", "Head × Pairwise Rank")
            source_role = attention.pairwise_source_role.weight.detach().float().cpu().numpy()
            target_role = attention.pairwise_target_role.weight.detach().float().cpu().numpy()
            role_labels = self.role_names if len(self.role_names) == source_role.shape[1] else [
                f"Semantic Feature {i}" for i in range(source_role.shape[1])]
            self.heatmap(axes[row, 2], source_role, f"Block {row + 1}: Source Role Projection",
                         "Semantic Feature", "Attention Head", role_labels)
            self.heatmap(axes[row, 3], target_role, f"Block {row + 1}: Target Role Projection",
                         "Semantic Feature", "Attention Head", role_labels)
            axes[row, 4].bar(np.arange(heads), attention.pairwise_scale.detach().float().cpu().numpy())
            axes[row, 4].set_title(f"Block {row + 1}: Pairwise Scale", loc="left")
            axes[row, 4].set_xlabel("Attention Head")
            axes[row, 4].set_ylabel("Learned Scale")
            axes[row, 4].set_xticks(np.arange(heads), labels=[f"Head {i + 1}" for i in range(heads)], rotation=90)

        figure.subplots_adjust(left=0.055, right=0.99, top=0.97, bottom=0.12, hspace=0.5, wspace=0.34)

        if show:
            plt.show()

        return figure


class LGPSAttentionProjectionWeights(LGPSVisualizer):
    title = "LGPS Query And Key Projection Weights"

    def plot(self, frames, model_path, *, device="cpu", threads=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        model, _ = _load_model(model_path, device, threads)
        figure, axes = plt.subplots(len(model.blocks), 2, figsize=(32, 6 * len(model.blocks)), squeeze=False)
        figure.suptitle(self.title, y=0.995)

        for row, block in enumerate(model.blocks):
            attention = block.attention

            for column, (name, projection) in enumerate((("Query", attention.q), ("Key", attention.k))):
                matrix = projection.weight.detach().float().cpu().numpy().reshape(
                    attention.num_heads * attention.head_dim, -1)
                self.heatmap(axes[row, column], matrix, f"Block {row + 1}: {name} Projection",
                             "Model Dimension", "Head × Head Dimension")

        figure.subplots_adjust(left=0.055, right=0.99, top=0.97, bottom=0.08, hspace=0.5, wspace=0.25)

        if show:
            plt.show()

        return figure


class LGPSTokenEncoderWeights(LGPSVisualizer):
    title = "LGPS Token Encoder Weights"

    def plot(self, frames, model_path, *, device="cpu", threads=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        model, _ = _load_model(model_path, device, threads)
        encoders = _encoder_weights(model)

        if not encoders:
            raise ValueError("The selected model has no input encoders to visualize")

        figure, axes = plt.subplots(len(encoders), 1, figsize=(36, 5.5 * len(encoders)), squeeze=False)
        figure.suptitle(self.title, y=0.995)

        for axis, (name, feature_names, weights) in zip(axes[:, 0], encoders):
            self.heatmap(axis, weights, f"{LGPSInputFeatureHeatmaps.encoder_titles[name]} Encoder",
                         "Input Feature Index", "Hidden Unit", [str(index) for index in range(len(feature_names))])

        figure.subplots_adjust(left=0.055, right=0.995, top=0.97, bottom=0.04, hspace=0.4)

        if show:
            plt.show()

        return figure


class LGPSOutputHeadWeights(LGPSVisualizer):
    title = "LGPS Output Head Weights"

    def plot(self, frames, model_path, *, device="cpu", threads=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        model, _ = _load_model(model_path, device, threads)
        input_layer = model.head[1]
        input_weights = np.concatenate((input_layer.weight.detach().float().cpu().numpy(),
                                        input_layer.bias.detach().float().cpu().numpy()[:, None]), axis=1)
        output_layer = model.head[-1]
        output_weights = np.concatenate((output_layer.weight.detach().float().cpu().numpy(),
                                         output_layer.bias.detach().float().cpu().numpy()[:, None]), axis=1)
        names = [f"Blue Goal Bucket {i + 1}" for i in range(model.config.targets_per_team)]
        names += [f"Orange Goal Bucket {i + 1}" for i in range(model.config.targets_per_team)]

        if not model.config.two_way_prediction:
            names += ["Neither Goal"]

        names = names[:output_weights.shape[0]]
        figure, axes = plt.subplots(2, 1, figsize=(34, 22), squeeze=False)
        figure.suptitle(self.title, y=0.995)
        first, final = axes[:, 0]
        self.heatmap(first, input_weights, "Pooled + Context Projection (Bias In Last Column)",
                     "Pooled/Context Hidden Dimension + Bias", "Output Hidden Unit")
        first_positions = list(range(0, input_weights.shape[1] - 1, 32)) + [input_weights.shape[1] - 1]
        first.set_xticks(first_positions, labels=[str(value) for value in first_positions[:-1]] + ["Bias"])
        first.axvline(model.config.d_model - 0.5, color="#202a35", linewidth=0.8)
        first.text(model.config.d_model - 0.5, -0.08, "Pooled | Context →", transform=first.get_xaxis_transform(),
                   ha="center", va="top", fontsize=10)
        self.heatmap(final, output_weights, "Final Goal-Probability Projection (Bias In Last Column)",
                     "Output Hidden Dimension + Bias", "Goal-Probability Output")
        final.set_yticks(np.arange(len(names)), labels=names, fontsize=11)
        positions = list(range(0, output_weights.shape[1] - 1, 32)) + [output_weights.shape[1] - 1]
        final.set_xticks(positions, labels=[str(value) for value in positions[:-1]] + ["Bias"], fontsize=10)
        figure.subplots_adjust(left=0.13, right=0.99, top=0.94, bottom=0.08, hspace=0.5)

        if show:
            plt.show()

        return figure


def _encoder_weights(model):
    feature_sets = {
        "ball": BALL_FEATURES,
        "player": PLAYER_FEATURES,
        "goal": GOAL_FEATURES,
        "big_boost": BOOST_FEATURES,
        "small_boost": BOOST_FEATURES,
        "context": CONTEXT_FEATURES,
    }
    encoders = []

    for name, module in (
        ("ball", model.ball_stream),
        ("player", model.player_stream),
        ("goal", model.goal_stream),
        ("big_boost", model.big_boost_stream),
        ("small_boost", model.small_boost_stream),
        ("context", model.context_token),
    ):
        if module is None:
            continue

        weights = module.network[0].weight.detach().float().cpu().numpy()
        names = list(feature_sets[name])

        if module.missing_aware:
            names.extend(f"{feature} (Missing)" for feature in feature_sets[name])

        if weights.shape[1] != len(names):
            raise ValueError(f"Feature names do not match the {name} encoder input width")

        encoders.append((name, names, weights))

    return encoders


class LGPSGoalProbabilitiesTimeline(LGPSVisualizer):
    title = "Live Goal Probabilities"
    output_directory = Path(__file__).resolve().parents[1] / "results" / "lgps"
    line_colors = ("#3b9eff", "#ff851f", "#e5e7eb")
    probability_labels = (
        "Blue Goal in Next 10 Seconds",
        "Orange Goal in Next 10 Seconds",
        "Neither Goal in the Next 10 Seconds",
    )

    def metadata_value(self, frame, column_names):
        for column in column_names:
            if column not in frame.columns:
                continue

            value = frame.get_column(column)[0]

            if value is not None and str(value).strip():
                return str(value).strip()

        return None

    def replay_title(self, replay):
        title = self.metadata_value(
            replay,
            ("match_title", "replay_title", "event_title", "match_name", "replay_name"),
        )

        if title:
            return f"{title} {self.title}"

        blue_team = self.metadata_value(replay, ("blue_team_name",))
        orange_team = self.metadata_value(replay, ("orange_team_name",))
        game = self.metadata_value(replay, ("game_number", "series_game_number", "game_id"))
        replay_date = self.metadata_value(replay, ("replay_datetime", "replay_date"))
        parts = []

        if blue_team and orange_team:
            parts.append(f"{blue_team} vs {orange_team}")

        if game:
            parts.append(game if game.lower().startswith("g") else f"G{game}")

        if replay_date:
            parts.append(replay_date)

        if not parts:
            parts.append(str(replay.get_column("replay_id")[0]))

        return f"{' '.join(parts)} {self.title}"

    def plot_replay(self, replay, replay_id, output_dir, *, show):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        elapsed_column = next((
            name for name in ("frame_seconds_elapsed", "seconds_elapsed")
            if name in replay.columns
        ), None)

        if elapsed_column is not None:
            elapsed = replay.get_column(elapsed_column).cast(pl.Float64).to_numpy()
        elif "game_seconds_remaining" in replay.columns:
            remaining = replay.get_column("game_seconds_remaining").cast(pl.Float64).to_numpy()
            elapsed = 300.0 - remaining
        elif "frame_number" in replay.columns:
            elapsed = replay.get_column("frame_number").cast(pl.Float64).to_numpy() / 30.0
        else:
            raise ValueError("Replay frames must include elapsed time, game time remaining, or frame_number")

        probabilities = replay.select(lgps_inference.PROBABILITY_COLUMNS).to_numpy()
        valid = np.isfinite(elapsed) & np.isfinite(probabilities).all(axis=1)

        if not valid.any():
            raise ValueError(f"Replay {replay_id} has no finite probability estimates to plot")

        elapsed = elapsed[valid]
        probabilities = probabilities[valid]
        order = np.argsort(elapsed, kind="stable")
        elapsed = elapsed[order]
        probabilities = probabilities[order]
        figure, axis = plt.subplots(figsize=(17.92, 7.35), dpi=100)
        figure.patch.set_facecolor("#111827")
        axis.set_facecolor("#182235")

        for index, label in enumerate(self.probability_labels):
            axis.plot(elapsed, probabilities[:, index], color=self.line_colors[index],
                      linewidth=1.35, label=label)

        axis.set_xlim(0.0, float(elapsed.max()))
        axis.set_ylim(0.0, 1.0)
        axis.xaxis.set_major_locator(MultipleLocator(50.0))
        axis.yaxis.set_major_locator(MultipleLocator(0.2))
        axis.set_xlabel("Time in Match", color="#d1d5db", fontsize=13)
        axis.set_ylabel("Probability", color="#d1d5db", fontsize=13)
        axis.set_title(self.replay_title(replay), color="#d1d5db", fontsize=17, pad=10)
        axis.tick_params(colors="#cbd5e1", labelsize=11)
        axis.grid(True, color="#334155", linewidth=0.65, alpha=0.8)

        for spine in axis.spines.values():
            spine.set_color("#64748b")

        legend = axis.legend(loc="lower left", frameon=True, fontsize=11,
                             facecolor="#111827", edgecolor="#cbd5e1", framealpha=0.75)

        for label in legend.get_texts():
            label.set_color("#e5e7eb")

        figure.tight_layout(pad=1.0)
        safe_replay_id = re.sub(r"[^\w.-]+", "_", str(replay_id)).strip("._") or "replay"
        output_path = output_dir / f"lgps_goal_probabilities_{safe_replay_id}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, facecolor=figure.get_facecolor())

        if show:
            plt.show()

        plt.close(figure)
        return output_path

    def plot(
        self,
        frames: pl.DataFrame | str | Path,
        model_path: str | Path = lgps_inference.MODEL_PATH,
        *,
        output_dir: str | Path | None = None,
        sample_rate: float = 10.0,
        device: str = "cpu",
        threads: int | None = None,
        show: bool = True,
        fix_context: bool = False,
    ):
        output_directory = self.output_directory if output_dir is None else Path(output_dir)
        model_path = Path(model_path).expanduser().resolve()
        original_posix_path = pathlib.PosixPath

        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath

        try:
            with TemporaryDirectory(prefix="lgps_goal_probabilities_") as temporary:
                temporary_path = Path(temporary)

                if isinstance(frames, pl.DataFrame):
                    source = _frame_source(frames, temporary_path)
                else:
                    source = Path(frames).expanduser().resolve()

                    if not source.exists():
                        raise FileNotFoundError(source)

                scored_paths = lgps_inference.apply_lgps(
                    source,
                    model_path=model_path,
                    output_dir=temporary_path / "scored",
                    device=device,
                    threads=threads,
                    sample_rate=sample_rate,
                    fix_context=fix_context,
                )

                if not scored_paths:
                    raise ValueError("LGPS produced no estimates for the supplied frames")

                output_paths = []
                source_frame = lgps_inference._read_frame(source) if source.is_file() else None
                source_replay_id = (
                    str(source_frame.get_column("replay_id")[0])
                    if source_frame is not None and "replay_id" in source_frame.columns
                    and source_frame.get_column("replay_id").n_unique() == 1
                    else None
                )

                for scored_path in scored_paths:
                    scored = lgps_inference._read_frame(scored_path)

                    if source_replay_id is not None:
                        metadata = [pl.lit(source_replay_id).alias("replay_id")]
                        for name in ("blue_team_name", "orange_team_name"):
                            if name in source_frame.columns:
                                metadata.append(pl.lit(source_frame.get_column(name)[0]).alias(name))
                        scored = scored.with_columns(metadata)

                    if "replay_id" not in scored.columns:
                        raise ValueError("Replay frames must include replay_id to name the probability plot")

                    for replay in scored.partition_by("replay_id", maintain_order=True):
                        replay_id = replay.get_column("replay_id")[0]
                        output_paths.append(self.plot_replay(
                            replay, replay_id, output_directory, show=False,
                        ))

                if show:
                    import matplotlib.pyplot as plt

                    plt.show()

                return output_paths
        finally:
            pathlib.PosixPath = original_posix_path


class LGPSSummaryGraph(LGPSVisualizer):
    output_directory = Path(__file__).resolve().parents[1] / "visualizations" / "lgps"
    background_color = "#101722"
    text_color = "#f3f6fa"
    edge_color = "#38475c"
    query_color = "#3b9eff"
    key_color = "#ff758f"
    value_color = "#66d9b5"
    embedding_color = "#9b82ff"
    probability_colors = ("#3b9eff", "#ff851f", "#e5e7eb")

    @staticmethod
    def team_names(frame):
        return tuple(
            str(frame[name][0]).strip() if name in frame.columns and frame[name][0]
            else fallback
            for name, fallback in (("blue_team_name", "Blue"),
                                   ("orange_team_name", "Orange"))
        )

    def frame_tokens(self, frame, values, model):
        blue_name, orange_name = self.team_names(frame)
        tokens = _tokens(frame, values, model, _player_name_map(frame))
        for token in tokens:
            if token["kind"] == "goal":
                token["label"] = f"{(blue_name, orange_name)[token['team']]} Goal"
            elif token["kind"] == "player" and token["label"] == "Player":
                slot = token["index"] - (3 if token["team"] == 1 else 0)
                token["label"] = f"{(blue_name, orange_name)[token['team']]} Player {slot}"
        return tokens

    def draw_arrow(self, figure, start, end, *, color=None, bend=0.0, alpha=0.8, zorder=1):
        from matplotlib.patches import FancyArrowPatch

        figure.add_artist(FancyArrowPatch(
            start,
            end,
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.8,
            color=color or self.text_color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={bend}",
            zorder=zorder,
        ))

    def draw_curve(self, figure, start, control_one, control_two, end, *, color, alpha, arrow=False,
                   dashed=False):
        from matplotlib.path import Path as PlotPath
        from matplotlib.patches import PathPatch

        path = PlotPath(
            (start, control_one, control_two, end),
            (PlotPath.MOVETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4),
        )
        patch = PathPatch(
            path,
            transform=figure.transFigure,
            facecolor="none",
            edgecolor=color,
            linewidth=0.8,
            alpha=alpha,
            capstyle="round",
            joinstyle="round",
            zorder=0.5,
        )
        if arrow:
            from matplotlib.patches import FancyArrowPatch

            patch = FancyArrowPatch(
                path=path,
                transform=figure.transFigure,
                arrowstyle="-|>",
                linestyle="--" if dashed else "-",
                mutation_scale=7,
                linewidth=0.8,
                color=color,
                alpha=alpha,
                zorder=0.5,
            )
        figure.add_artist(patch)

    def draw_box(self, figure, bounds, *, edgecolor, linewidth=0.8):
        from matplotlib.patches import Rectangle

        figure.add_artist(Rectangle(
            (bounds[0], bounds[1]), bounds[2], bounds[3],
            transform=figure.transFigure,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=0,
        ))

    def draw_frame_state(self, axis, tokens, scoreboard):
        from analyzerl_boxcars.animate import _draw_boost_pads_2d, _draw_field

        axis.set_facecolor(self.background_color)
        _draw_field(axis)
        _draw_boost_pads_2d(axis)
        axis.set_xlim(4096, -4096)
        axis.set_ylim(-6000, 6000)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()

        scoreboard_artist = axis.text(
            0.01, 0.99, scoreboard, transform=axis.transAxes,
            va="top", ha="left", color=self.text_color, zorder=8,
        )

        for token in tokens:
            position = np.asarray(token["xy"], dtype=np.float64)
            if position.shape[0] < 2 or not np.isfinite(position[:2]).all():
                continue
            if token["kind"] == "player" and token["label"] == "Empty player slot":
                continue

            kind = token["kind"]
            size = {"ball": 45, "player": 55, "goal": 0,
                    "big_boost": 24, "small_boost": 12}[kind]
            if size:
                color = TEAM_COLORS[token["team"]] if kind == "player" else TYPE_COLORS[kind]
                axis.scatter(
                    position[0], position[1], s=size, c=[color],
                    edgecolors=self.background_color, linewidths=0.6, zorder=5,
                )
            if kind == "player":
                axis.annotate(
                    token["label"], (position[0], position[1]), xytext=(0, 5),
                    textcoords="offset points", color=self.text_color,
                    ha="center", va="bottom", rotation=0, zorder=6,
                )

        return scoreboard_artist

    def draw_token_flow(self, figure, tokens, embeddings, queries, keys, values_by_token):
        from matplotlib.patches import Ellipse

        retained = []
        for index, token in enumerate(tokens):
            position = np.asarray(token["xy"], dtype=np.float64)
            if position.shape[0] < 2 or not np.isfinite(position[:2]).all():
                continue
            if token["kind"] == "player" and token["label"] == "Empty player slot":
                continue
            retained.append((index, token))

        tokens = [token for _, token in retained]
        entity_indices = [index for index, _ in retained]
        if len(entity_indices) != len(tokens):
            raise RuntimeError("Token labels do not match the model entity axis")

        top, bottom = 0.900, 0.100
        row_height = (top - bottom) / max(len(tokens), 1)
        row_centers = top - (np.arange(len(tokens)) + 0.5) * row_height
        embedding_values = embeddings[entity_indices]
        embedding_limit = max(float(np.nanpercentile(np.abs(embedding_values), 99)), 1e-6)
        q_norms = np.linalg.norm(queries, axis=-1)
        k_norms = np.linalg.norm(keys, axis=-1)
        v_norms = np.linalg.norm(values_by_token, axis=-1)
        projection_columns = (
            (0.395, q_norms, self.query_color),
            (0.418, k_norms, self.key_color),
            (0.441, v_norms, self.value_color),
        )

        for row_index, (entity_index, token) in enumerate(retained):
            y = float(row_centers[row_index])
            team = token["team"]
            color = TEAM_COLORS[team] if token["kind"] == "player" and team in (0, 1) else TYPE_COLORS[token["kind"]]
            figure.text(
                0.316, y, token["label"], color=color,
                ha="right", va="center",
            )
            figure.add_artist(Ellipse(
                (0.325, y),
                width=2 * 0.0036 * figure.get_figheight() / figure.get_figwidth(),
                height=2 * 0.0036,
                transform=figure.transFigure,
                facecolor=color,
                edgecolor=self.background_color,
                linewidth=0.25,
                zorder=3,
            ))
            figure.add_artist(Ellipse(
                (0.371, y),
                width=0.022 * figure.get_figheight() / figure.get_figwidth(),
                height=0.022,
                transform=figure.transFigure,
                facecolor=self.embedding_color,
                edgecolor=self.background_color,
                linewidth=0.35,
                zorder=4,
            ))
            for x, norm_values, node_color in projection_columns:
                maximum = max(float(np.nanmax(norm_values)), 1e-8)
                size = 8 + 18 * float(np.clip(norm_values[entity_index] / maximum, 0, 1))
                node_radius = 0.0022 + size / 10000
                figure.add_artist(Ellipse(
                    (x, y),
                    width=2 * node_radius * figure.get_figheight() / figure.get_figwidth(),
                    height=2 * node_radius,
                    transform=figure.transFigure,
                    facecolor=node_color,
                    edgecolor=self.background_color,
                    linewidth=0.25,
                    zorder=3,
                ))
            for x1, x2, line_color in (
                (0.372, 0.392, self.query_color),
                (0.372, 0.415, self.key_color),
                (0.372, 0.438, self.value_color),
            ):
                self.draw_arrow(
                    figure, (x1, y), (x2, y), color=line_color,
                    alpha=0.58,
                )

        embedding_axis = figure.add_axes((0.350, 0.127, 0.020, 0.746))
        embedding_axis.set_facecolor(self.background_color)
        embedding_axis.imshow(
            embedding_values,
            cmap="coolwarm",
            vmin=-embedding_limit,
            vmax=embedding_limit,
            interpolation="nearest",
            aspect="auto",
        )
        embedding_axis.set_axis_off()
        for x, label, color in (
            (0.395, "Q", self.query_color),
            (0.418, "K", self.key_color),
            (0.441, "V", self.value_color),
        ):
            figure.text(x, 0.930, label, color=color,
                        ha="center", va="bottom")

        return tokens, row_centers, entity_indices

    def draw_frame_to_tokens(self, figure, field_axis, tokens, row_centers):
        # Resolve the field's data transform after its aspect ratio has settled.
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for token, y in zip(tokens, row_centers):
            position = np.asarray(token["xy"], dtype=np.float64)
            start = to_figure.transform(field_axis.transData.transform(position[:2]))
            color = (TEAM_COLORS[token["team"]]
                     if token["kind"] == "player" else TYPE_COLORS[token["kind"]])
            self.draw_arrow(
                figure, tuple(start), (0.322, float(y)), color=color,
                bend=0.10 if start[1] < y else -0.10,
                alpha=0.42,
            )

    def draw_context_path(self, figure, scoreboard_artist):
        from matplotlib.path import Path as PlotPath
        from matplotlib.patches import Ellipse, FancyArrowPatch

        y = 0.075
        figure.text(0.316, y, "Context", color=self.embedding_color,
                    ha="right", va="center")
        for x, radius in ((0.325, 0.0036), (0.371, 0.011)):
            figure.add_artist(Ellipse(
                (x, y),
                width=2 * radius * figure.get_figheight() / figure.get_figwidth(),
                height=2 * radius,
                transform=figure.transFigure,
                facecolor=self.embedding_color,
                edgecolor=self.background_color,
                linewidth=0.35,
                zorder=4,
            ))
        bbox = scoreboard_artist.get_window_extent(renderer=figure.canvas.get_renderer())
        start = figure.transFigure.inverted().transform(
            (bbox.x1 + 2, bbox.y0 + bbox.height / 2)
        )
        path = PlotPath(
            (start, (0.210, start[1]), (0.210, 0.030), (0.310, 0.030), (0.322, y)),
            (PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO,
             PlotPath.LINETO, PlotPath.LINETO),
        )
        figure.add_artist(FancyArrowPatch(
            path=path, transform=figure.transFigure, arrowstyle="-|>",
            mutation_scale=7, linewidth=0.8, color=self.embedding_color,
            alpha=0.55, zorder=1,
        ))
        self.draw_arrow(figure, (0.328, y), (0.368, y),
                        color=self.embedding_color, alpha=0.65)
        self.draw_curve(
            figure,
            (0.372, y), (0.530, 0.010), (0.790, 0.020), (0.895, 0.360),
            color=self.embedding_color, alpha=0.5, arrow=True, dashed=True,
        )

    def draw_attention_flows(self, figure, row_centers, attention_axis):
        # Use the rendered image transform so each Q arrow lands on its query
        # row and each K arrow lands above its matching key column.
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()

        for row, y in enumerate(row_centers):
            query_edge = to_figure.transform(
                attention_axis.transData.transform((-0.5, row))
            )
            key_edge = to_figure.transform(
                attention_axis.transData.transform((row, -0.5))
            )
            self.draw_arrow(
                figure, (0.398, float(y)),
                (float(query_edge[0]), float(query_edge[1])),
                color=self.query_color, alpha=0.34, zorder=3,
            )

            self.draw_curve(
                figure,
                (0.421, float(y)),
                (0.450, float(y)),
                (float(key_edge[0]), float(key_edge[1]) + 0.060),
                (float(key_edge[0]), float(key_edge[1]) + 0.016),
                color=self.key_color,
                alpha=0.24,
            )
            self.draw_arrow(
                figure,
                (float(key_edge[0]), float(key_edge[1]) + 0.016),
                (float(key_edge[0]), float(key_edge[1])),
                color=self.key_color, alpha=0.45, zorder=3,
            )

            output_y = 0.865 - 0.730 * row / max(len(row_centers) - 1, 1)
            self.draw_curve(
                figure,
                (0.444, float(y)),
                (0.520, float(y) + 0.025),
                (0.660, output_y + 0.025),
                (0.734, output_y),
                color=self.value_color,
                alpha=0.30,
            )

    def draw_attention_map(self, figure, matrix):
        axis = figure.add_axes((0.455, 0.100, 0.238, 0.800))
        axis.set_zorder(2)
        axis.set_facecolor(self.background_color)
        strengths = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
        axis.imshow(strengths, cmap="plasma", vmin=0.0,
                    vmax=max(float(np.nanmax(strengths)), 1e-8),
                    interpolation="nearest", aspect="equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(self.edge_color)
            spine.set_linewidth(0.5)
        return axis

    def draw_mlp(self, figure, token_indices, hidden_activations, output_activations):
        from matplotlib.patches import Ellipse

        hidden_norms = np.linalg.norm(hidden_activations, axis=-1)
        output_norms = np.linalg.norm(output_activations, axis=-1)
        hidden_scale = max(float(np.nanmax(hidden_norms)), 1e-8)
        output_scale = max(float(np.nanmax(output_norms)), 1e-8)
        first_x, second_x = 0.734, 0.808
        self.draw_box(figure, (0.715, 0.100, 0.110, 0.800), edgecolor=self.embedding_color)
        for row_index, token_index in enumerate(token_indices):
            y = 0.865 - 0.730 * row_index / max(len(token_indices) - 1, 1)
            first_size = 8 + 22 * float(np.clip(hidden_norms[token_index] / hidden_scale, 0, 1))
            second_size = 8 + 22 * float(np.clip(output_norms[token_index] / output_scale, 0, 1))
            for x, size, color in ((first_x, first_size, self.value_color),
                                   (second_x, second_size, self.embedding_color)):
                node_radius = 0.002 + size / 9500
                figure.add_artist(Ellipse(
                    (x, y),
                    width=2 * node_radius * figure.get_figheight() / figure.get_figwidth(),
                    height=2 * node_radius,
                    transform=figure.transFigure,
                    facecolor=color,
                    edgecolor=self.text_color,
                    linewidth=0.2,
                    zorder=3,
                ))
            self.draw_arrow(figure, (first_x + 0.006, y), (second_x - 0.006, y),
                            color=self.embedding_color, alpha=0.5)

    def draw_readout(self, figure, model, probabilities, team_names):
        from matplotlib.patches import Rectangle, PathPatch
        from matplotlib.path import Path as PlotPath

        self.draw_box(figure, (0.845, 0.350, 0.030, 0.300), edgecolor=self.value_color)
        self.draw_box(figure, (0.895, 0.350, 0.030, 0.300), edgecolor=self.value_color)
        figure.text(0.860, 0.500, "Query Pool", color=self.value_color,
                    ha="center", va="center")
        figure.text(0.910, 0.500, "Head", color=self.value_color,
                    ha="center", va="center")
        self.draw_arrow(figure, (0.825, 0.500), (0.845, 0.500), color=self.embedding_color)
        self.draw_arrow(figure, (0.875, 0.500), (0.895, 0.500), color=self.value_color)
        self.draw_arrow(figure, (0.925, 0.500), (0.940, 0.500), color=self.value_color)

        brace_points = [
            (0.940, 0.755),
            (0.950, 0.755), (0.948, 0.690), (0.948, 0.625),
            (0.948, 0.555), (0.948, 0.515), (0.954, 0.500),
            (0.948, 0.485), (0.948, 0.445), (0.948, 0.375),
            (0.948, 0.310), (0.950, 0.245), (0.940, 0.245),
        ]
        brace_path = PlotPath(brace_points, [PlotPath.MOVETO] + [PlotPath.CURVE4] * 12)
        figure.add_artist(PathPatch(brace_path, transform=figure.transFigure,
                                    facecolor="none", edgecolor=self.value_color,
                                    linewidth=0.8, zorder=2))
        horizon = model.config.horizon_seconds
        labels = (f"{team_names[0]} Goal In Next {horizon:g} Seconds",
                  f"{team_names[1]} Goal In Next {horizon:g} Seconds",
                  f"Neither Goal in the Next {horizon:g} Seconds")
        for index, (label, probability, color) in enumerate(
            zip(labels, probabilities, self.probability_colors)
        ):
            y = 0.745 - index * 0.245
            self.draw_arrow(figure, (0.948 if index != 1 else 0.954, y),
                            (0.959, y), color=color, alpha=0.8)
            figure.text(0.960, y + 0.024, label, color=self.text_color,
                        fontsize=8, ha="left", va="bottom")
            figure.add_artist(Rectangle(
                (0.960, y - 0.010), 0.035, 0.018,
                transform=figure.transFigure,
                facecolor="#263347", edgecolor="none", zorder=2,
            ))
            figure.add_artist(Rectangle(
                (0.960, y - 0.010), 0.035 * float(probability), 0.018,
                transform=figure.transFigure,
                facecolor=color, edgecolor="none", zorder=3,
            ))
            figure.text(0.960, y - 0.025, f"{probability:.1%}", color=color,
                        ha="left", va="center")

    def frame_metadata(self, frame, frame_index):
        from analyzerl_boxcars.animate import _clock_label

        frame_number = (
            int(frame.get_column("frame_number")[0])
            if "frame_number" in frame.columns else frame_index
        )
        elapsed = (
            float(frame.get_column("frame_seconds_elapsed")[0])
            if "frame_seconds_elapsed" in frame.columns
            else frame_number / 30.0
        )
        score_columns = ("blue_team_score", "blue_score")
        blue_score = next((
            float(frame.get_column(name).cast(pl.Float64).fill_null(0.0)[0])
            for name in score_columns if name in frame.columns
        ), 0.0)
        orange_score = next((
            float(frame.get_column(name).cast(pl.Float64).fill_null(0.0)[0])
            for name in ("orange_team_score", "orange_score") if name in frame.columns
        ), 0.0)
        blue_name, orange_name = self.team_names(frame)
        return (f"{blue_name} {blue_score:.0f} - {orange_score:.0f} {orange_name}"
                f" | frame {frame_number} | {_clock_label(elapsed)}")

    def plot(
        self,
        frames: pl.DataFrame | str | Path,
        model_path: str | Path = lgps_inference.MODEL_PATH,
        *,
        replay_id: str | None = None,
        frame_index: int | None = None,
        output_dir: str | Path | None = None,
        sample_rate: float = 10.0,
        device: str = "cpu",
        threads: int | None = None,
        show: bool = True,
    ):
        import matplotlib.pyplot as plt

        model_path = Path(model_path).expanduser().resolve()
        original_posix_path = pathlib.PosixPath
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        try:
            with TemporaryDirectory(prefix="lgps_summary_graph_") as temporary:
                temporary_path = Path(temporary)
                source = _frame_source(frames, temporary_path)
                probability_columns = (
                    "latent_blue_goal_probability",
                    "latent_orange_goal_probability",
                    "latent_neither_goal_probability",
                )
                source_is_scored = (
                    source.is_file()
                    and set(probability_columns).issubset(
                        lgps_inference._read_frame(source).columns
                    )
                )
                scored_root = temporary_path / "scored"
                scored_paths = (
                    [source]
                    if source_is_scored
                    else lgps_inference.apply_lgps(
                        source, model_path=model_path, output_dir=scored_root,
                        device=device, threads=threads, sample_rate=sample_rate,
                    )
                )
                if not scored_paths:
                    raise ValueError("LGPS produced no estimates for the supplied frames")

                selected_replay = None
                selected_source = None
                for scored_path in scored_paths:
                    scored = lgps_inference._read_frame(scored_path)
                    selected_source = (
                        source if source_is_scored or source.is_file()
                        else source / scored_path.relative_to(scored_root)
                    )
                    if "replay_id" not in scored.columns:
                        if replay_id is None:
                            selected_replay = scored
                            break
                        continue
                    selected_replay = next((
                        replay for replay in scored.partition_by("replay_id", maintain_order=True)
                        if replay_id is None or str(replay["replay_id"][0]) == str(replay_id)
                    ), None)
                    if selected_replay is not None:
                        break
                if selected_replay is None or selected_replay.is_empty():
                    raise ValueError(f"No inference frames found for replay_id={replay_id!r}")

                selected_index = selected_replay.height // 2 if frame_index is None else frame_index
                if selected_index < 0:
                    selected_index += selected_replay.height
                if selected_index < 0 or selected_index >= selected_replay.height:
                    raise IndexError(f"frame_index must be in [0, {selected_replay.height - 1}]")
                frame = selected_replay.slice(selected_index, 1)
                frame_number = (int(frame["frame_number"][0])
                                if "frame_number" in frame.columns else selected_index)
                source_frames = lgps_inference._read_frame(selected_source)
                if replay_id is None and "replay_id" in selected_replay.columns:
                    replay_id = selected_replay.get_column("replay_id")[0]
                raw_replay = source_frames
                if replay_id is not None and "replay_id" in raw_replay.columns:
                    raw_replay = raw_replay.filter(pl.col("replay_id") == replay_id)
                raw_frame = raw_replay
                if "frame_number" in raw_replay.columns and "frame_number" in frame.columns:
                    frame_number = frame.get_column("frame_number")[0]
                    matching_frame = raw_replay.filter(pl.col("frame_number") == frame_number)
                    if not matching_frame.is_empty():
                        raw_frame = matching_frame
                source_row = raw_frame.row(0, named=True) if not raw_frame.is_empty() else {}
                metadata_names = {
                    "blue_team_name", "orange_team_name",
                }
                metadata_names.update(
                    name for name in source_row
                    if name.startswith("player_") and name.endswith("_name")
                )
                metadata_updates = [
                    pl.lit(source_row[name]).alias(name)
                    for name in metadata_names
                    if name in source_row and source_row[name] is not None
                ]
                if metadata_updates:
                    frame = frame.with_columns(metadata_updates)

                model, torch = _load_model(model_path, device, threads)
                input_names = tuple(
                    name for name in lgps_inference.INPUT_NAMES
                    if name not in ("big_boosts", "small_boosts") or name in model.model_entities
                )
                player_slots = lgps_inference._player_slots(frame)
                input_feature_columns = [
                    *(f"player_{slot}_{feature}"
                      for slot in player_slots if slot is not None
                      for feature in lgps_inference.PLAYER_FEATURES),
                    *lgps_inference.BALL_FEATURES,
                    *lgps_inference.CONTEXT_FEATURES,
                ]
                if "big_boosts" in input_names:
                    input_feature_columns.extend(lgps_inference.BIG_BOOST_COLUMNS)
                if "small_boosts" in input_names:
                    input_feature_columns.extend(lgps_inference.SMALL_BOOST_COLUMNS)
                missing_input_columns = [
                    name for name in input_feature_columns if name not in frame.columns
                ]
                if missing_input_columns:
                    player_prefix = "player_"
                    frame = frame.with_columns(
                        pl.lit(float("nan") if name.startswith(player_prefix) else 0.0)
                        .cast(pl.Float32).alias(name)
                        for name in missing_input_columns
                    )
                values = lgps_inference._model_inputs(frame, input_names)
                tensors = {name: torch.from_numpy(value).to(device) for name, value in values.items()}
                embedded = []
                attention_maps = []
                mlp_hidden = []
                mlp_output = []
                first_block = model.blocks[0]

                def capture_embedding(module, inputs):
                    if not embedded:
                        embedded.append(inputs[0].detach().float().cpu().numpy()[0])

                def capture_attention(module, inputs, output):
                    if not attention_maps:
                        attention_maps.append(module.last_attention.detach().float().cpu().numpy()[0])

                def capture_hidden(module, inputs, output):
                    if not mlp_hidden:
                        mlp_hidden.append(output.detach().float().cpu().numpy()[0])

                def capture_mlp_output(module, inputs, output):
                    if not mlp_output:
                        mlp_output.append(output.detach().float().cpu().numpy()[0])

                hooks = [
                    first_block.register_forward_pre_hook(capture_embedding),
                    first_block.attention.register_forward_hook(capture_attention),
                    first_block.ffn[0].register_forward_hook(capture_hidden),
                    first_block.ffn[3].register_forward_hook(capture_mlp_output),
                ]
                first_block.attention.capture_attention = True
                try:
                    with torch.inference_mode():
                        logits = model(**tensors)
                finally:
                    for hook in hooks:
                        hook.remove()

                if not embedded or not attention_maps or not mlp_hidden or not mlp_output:
                    raise RuntimeError("Could not capture the first transformer block activations")
                stream = torch.from_numpy(embedded[0]).unsqueeze(0).to(device)
                normalized = first_block.norm1(stream)
                attention = first_block.attention
                batch, entities, _ = normalized.shape
                head_count = model.config.num_heads
                head_width = model.config.d_model // head_count
                query = attention.q(normalized).view(batch, entities, head_count, head_width)[0].detach().cpu().numpy()
                key = attention.k(normalized).view(batch, entities, head_count, head_width)[0].detach().cpu().numpy()
                value = attention.v(normalized).view(batch, entities, head_count, head_width)[0].detach().cpu().numpy()
                selected_head = 0
                matrix = attention_maps[0][selected_head]
                probabilities = lgps_inference._probabilities(
                    logits.float().cpu().numpy(), model.config.targets_per_team,
                    model.config.horizon_seconds, model.config.two_way_prediction,
                )[0]
                scoreboard = self.frame_metadata(frame, selected_index)
                team_names = self.team_names(frame)
                frame_tokens = self.frame_tokens(frame, values, model)

                figure = plt.figure(figsize=(52, 15.5), facecolor=self.background_color)
                figure.suptitle("LGPS Architectural Summary", color=self.text_color,
                                fontsize=19, y=0.985)
                for x, label in ((0.360, "Embedding"),
                                 (0.574, "Attention"),
                                 (0.770, "MLP")):
                    figure.text(x, 0.935, label, color=self.text_color,
                                ha="center", va="bottom", fontsize=13)
                field_axis = figure.add_axes((0.015, 0.055, 0.181, 0.890))
                scoreboard_artist = self.draw_frame_state(field_axis, frame_tokens, scoreboard)
                tokens, row_centers, token_indices = self.draw_token_flow(
                    figure, frame_tokens, embedded[0],
                    query[:, selected_head], key[:, selected_head], value[:, selected_head],
                )
                self.draw_frame_to_tokens(figure, field_axis, tokens, row_centers)
                matrix = matrix[np.ix_(token_indices, token_indices)]
                attention_axis = self.draw_attention_map(figure, matrix)
                self.draw_attention_flows(figure, row_centers, attention_axis)
                self.draw_mlp(
                    figure, token_indices, mlp_hidden[0], mlp_output[0],
                )
                self.draw_readout(figure, model, probabilities, team_names)
                self.draw_arrow(figure, (0.693, 0.50), (0.715, 0.50), color=self.embedding_color)
                if "context" in model.model_entities:
                    self.draw_context_path(figure, scoreboard_artist)
                output_directory = self.output_directory if output_dir is None else Path(output_dir)
                output_directory.mkdir(parents=True, exist_ok=True)
                safe_replay_id = re.sub(r"[^\w.-]+", "_", str(replay_id or source.stem)).strip("._") or "replay"
                output_path = output_directory / (
                    f"lgps_architectural_summary_{safe_replay_id}_{frame_number}.png"
                )
                figure.savefig(output_path, dpi=100, facecolor=figure.get_facecolor())
                if show:
                    plt.show()
                return figure
        finally:
            pathlib.PosixPath = original_posix_path


VIZ_TYPES: dict[str, type[LGPSVisualizer]] = {
    "attention_graph": LGPSAttentionGraph,
    "goal_probabilities_timeline": LGPSGoalProbabilitiesTimeline,
    "summary_graph": LGPSSummaryGraph,
    "input_weight_heatmaps": LGPSInputFeatureHeatmaps,
    "input_weight_magnitudes": LGPSWeightMagnitudes,
    "geometry_bias_weights": LGPSGeometryBiasWeights,
    "pairwise_state_weights": LGPSPairwiseStateWeights,
    "attention_projection_weights": LGPSAttentionProjectionWeights,
    "token_encoder_weights": LGPSTokenEncoderWeights,
    "output_head_weights": LGPSOutputHeadWeights,
}


def visualize_lgps(
    frames: pl.DataFrame | str | Path | None = None,
    model_path: str | Path = lgps_inference.MODEL_PATH,
    *,
    viz_type: str = "summary_graph",
    render_mode: str = "2d",
    **kwargs,
):
    try:
        visualizer = VIZ_TYPES[viz_type]()
    except KeyError as error:
        raise ValueError(f"viz_type must be one of {tuple(VIZ_TYPES)}") from error
    render_mode = render_mode.lower()
    if render_mode not in ("2d", "3d"):
        raise ValueError("render_mode must be '2d' or '3d'")
    if viz_type in ("attention_graph", "goal_probabilities_timeline", "summary_graph") and frames is None:
        raise ValueError(f"frames are required for viz_type={viz_type!r}")
    if viz_type != "attention_graph" and render_mode != "2d":
        raise ValueError("render_mode is only available for attention_graph")
    if viz_type == "attention_graph":
        kwargs["render_mode"] = render_mode
    return visualizer.plot(frames, model_path, **kwargs)
