from __future__ import annotations

import os
import numpy as np
import polars as pl
from contextlib import nullcontext
from pathlib import Path
from props import (
    BALL_FEATURES,
    BIG_BOOST_COLUMNS,
    BLUE_GOAL,
    BOOST_FEATURES,
    CONTEXT_FEATURES,
    GOAL_COLUMNS,
    GOAL_FEATURES,
    ORANGE_GOAL,
    PLAYER_FEATURES,
    SMALL_BOOST_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lgps_transformer.onnx"
OUTPUT_DIR = ROOT / "inference" / "output"
PROBABILITY_COLUMNS = (
    "latent_blue_goal_probability",
    "latent_orange_goal_probability",
    "latent_neither_goal_probability",
)
INPUT_NAMES = (
    "ball",
    "players",
    "goals",
    "entity_types",
    "entity_teams",
    "game_context",
    "big_boosts",
    "small_boosts",
)
TORCHSCRIPT_BATCH_SIZE = 256


def _input_paths(path: Path, output_dir: Path) -> list[Path]:
    path = path.expanduser().resolve()
    output_dir = output_dir.resolve()

    if path.is_file():
        files = [path] if path.suffix.lower() in (".csv", ".parquet") else []
    elif path.is_dir():
        files = sorted(file for file in path.rglob("*")
                       if file.is_file()
                       and file.suffix.lower() in (".csv", ".parquet")
                       and not file.resolve().is_relative_to(output_dir))
    else:
        raise FileNotFoundError(path)

    if not files:
        raise FileNotFoundError(f"No CSV or Parquet replay frames found under {path}")

    return files


def _read_frame(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=None) if path.suffix.lower() == ".csv" else pl.read_parquet(path)


def _columns(frame: pl.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    # Select columns together to avoid repeated Python and Polars calls for
    # every frame in a replay.
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing model input columns: {missing[:8]}")

    return frame.select(pl.col(name).cast(pl.Float32, strict=False).fill_null(
        -1.0 if name.endswith("_team_id") else 0.0)
                        for name in names).to_numpy().astype(np.float32, copy=False)


def _player_slots(frame: pl.DataFrame) -> tuple[int | None, ...]:
    if set(GOAL_COLUMNS).issubset(frame.columns):
        return (1, 2, 3, 4, 5, 6)

    teams = {0: [], 1: []}
    for name in frame.columns:
        if not name.startswith("player_") or not name.endswith("_team_id"):
            continue
        values = frame[name].drop_nulls().unique().to_list()
        if len(values) != 1 or int(values[0]) not in teams:
            raise ValueError(f"{name}: expected one stable blue/orange team id")
        teams[int(values[0])].append(int(name.split("_")[1]))

    if not teams[0] or not teams[1] or max(map(len, teams.values())) > 3:
        raise ValueError("Expected one to three roster players per team")

    return tuple(slot for team in (0, 1)
                 for slot in sorted(teams[team]) + [None] * (3 - len(teams[team])))


def _model_inputs(frame: pl.DataFrame, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    slots = _player_slots(frame)
    count = frame.height
    has_goals = set(GOAL_COLUMNS).issubset(frame.columns)
    big_boosts = "big_boosts" in names
    small_boosts = "small_boosts" in names
    columns = [*BALL_FEATURES]
    columns.extend(f"player_{slot}_{feature}" for slot in slots if slot is not None
                   for feature in PLAYER_FEATURES)
    if has_goals:
        columns.extend(GOAL_COLUMNS)
    columns.extend(CONTEXT_FEATURES)
    if big_boosts:
        columns.extend(BIG_BOOST_COLUMNS)
    if small_boosts:
        columns.extend(SMALL_BOOST_COLUMNS)
    columns.extend(f"player_{slot}_team_id" for slot in slots if slot is not None)
    matrix = _columns(frame, tuple(columns))
    offset = 0

    def take(width: int) -> np.ndarray:
        nonlocal offset
        values = matrix[:, offset:offset + width]
        offset += width
        return values

    ball = take(len(BALL_FEATURES))
    players = np.stack([take(len(PLAYER_FEATURES)) if slot is not None
                        else np.zeros((count, len(PLAYER_FEATURES)), np.float32)
                        for slot in slots], axis=1)
    if has_goals:
        goals = take(len(GOAL_COLUMNS)).reshape(count, 2, len(GOAL_FEATURES))
    else:
        goals = np.broadcast_to(np.stack((BLUE_GOAL, ORANGE_GOAL)),
                                (count, 2, len(GOAL_FEATURES))).copy()
        if "_rotate_180" in frame.columns:
            signs = np.where(frame["_rotate_180"].fill_null(False).to_numpy()[:, None, None],
                             (-1.0, -1.0, 1.0), (1.0, 1.0, 1.0)).astype(np.float32)
            goals[..., :3] *= signs
            goals[..., 6:9] *= signs

    context = take(len(CONTEXT_FEATURES))
    boost_values = take(len(BIG_BOOST_COLUMNS)).reshape(count, 6, len(BOOST_FEATURES)) if big_boosts else None
    small_values = take(len(SMALL_BOOST_COLUMNS)).reshape(count, 28, len(BOOST_FEATURES)) if small_boosts else None
    teams = np.stack([take(1)[:, 0] if slot is not None
                      else np.full(count, -1, np.float32)
                      for slot in slots], axis=1).astype(np.int64)
    teams = np.where((teams == 0) | (teams == 1), teams, 2)
    types = np.asarray([0] + [1] * 6 + [2] * 2
                       + [3] * (6 if big_boosts else 0)
                       + [3] * (28 if small_boosts else 0), dtype=np.int64)
    entity_teams = np.concatenate((np.full((count, 1), 2, np.int64), teams,
                                   np.tile(np.asarray([[0, 1]], np.int64), (count, 1)),
                                   np.full((count, len(types) - 9), 2, np.int64)), axis=1)
    values = {
        "ball": ball,
        "players": players,
        "goals": goals,
        "entity_types": types,
        "entity_teams": entity_teams,
        "game_context": context,
    }
    if big_boosts:
        values["big_boosts"] = boost_values
    if small_boosts:
        values["small_boosts"] = small_values

    return values


def _prepare_if_needed(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame

    if set(BALL_FEATURES).issubset(frame.columns) and set(CONTEXT_FEATURES).issubset(frame.columns):
        return frame

    if "frame_delta" not in frame.columns or "frame_number" not in frame.columns:
        raise ValueError("Expected model-ready LGPS columns or parsed replay frames")

    # Raw parsed replays need the repository's roster mapping and feature
    # preparation. Import it only for those inputs; prepared files stay light.
    from lgps.dataset import prepare_replay

    if "replay_id" not in frame.columns:
        return prepare_replay(frame, inference=True, target_sample_rate=30.0)

    return pl.concat([prepare_replay(group, inference=True, target_sample_rate=30.0)
                      for group in frame.partition_by("replay_id", maintain_order=True)],
                     how="vertical")


def _probabilities(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    values /= values.sum(axis=1, keepdims=True)
    per_team = (values.shape[1] - 1) // 2
    return np.column_stack((values[:, :per_team].sum(axis=1),
                            values[:, per_team:-1].sum(axis=1),
                            values[:, -1])).astype(np.float32, copy=False)


def apply_lgps(path: str | Path, model_path: str | Path = MODEL_PATH,
               output_dir: str | Path = OUTPUT_DIR, batch_size: int | None = None,
               threads: int | None = None, device: str = "cpu",
               precision: str = "fp32") -> list[Path]:
    """Write scored replay files and return their output paths."""
    path = Path(path)
    path = path.expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    threads = min(os.cpu_count() or 1, 8) if threads is None else threads
    if threads < 1:
        raise ValueError("threads must be positive")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    torchscript = model_path.name.endswith(".torchscript.pt")
    batch_size = (TORCHSCRIPT_BATCH_SIZE if torchscript else 512) if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be fp32 or fp16")
    if torchscript:
        import torch

        if precision == "fp16" and device != "cuda":
            raise ValueError("fp16 inference requires a CUDA TorchScript model")
        torch.set_num_threads(threads)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device not in ("cpu", "cuda"):
            raise ValueError("TorchScript device must be cpu or cuda")
        model = torch.jit.load(str(model_path), map_location="cpu").to(device).eval()
        names = tuple(argument.name for argument in model.forward.schema.arguments[1:])
    else:
        if device != "cpu":
            raise ValueError("The ONNX runner currently supports CPU only")
        if precision != "fp32":
            raise ValueError("fp16 inference requires a CUDA TorchScript model")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        session = ort.InferenceSession(str(model_path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        names = tuple(value.name for value in session.get_inputs())
    if not set(names).issubset(INPUT_NAMES):
        raise ValueError(f"Unsupported LGPS model inputs: {names}")

    outputs = []
    for source in _input_paths(path, output_dir):
        frame = _prepare_if_needed(_read_frame(source))
        if frame.is_empty():
            continue
        values = _model_inputs(frame, names)
        logits = []
        tensors = ({name: torch.from_numpy(value).to(device) for name, value in values.items()}
                   if torchscript else None)
        for start in range(0, frame.height, batch_size):
            end = min(start + batch_size, frame.height)
            if torchscript:
                arguments = []
                for name in names:
                    value = tensors[name]
                    if name != "entity_types":
                        value = value[start:end]
                        if end - start < batch_size:
                            value = torch.cat((value, value[-1:].expand(
                                batch_size - (end - start), *value.shape[1:])), dim=0)
                    arguments.append(value)
                autocast = (torch.autocast("cuda", dtype=torch.float16)
                            if precision == "fp16" else nullcontext())
                with torch.inference_mode(), autocast:
                    logits.append(model(*arguments)[:end - start].cpu().numpy())
            else:
                feed = {name: value if name == "entity_types" else value[start:end]
                        for name, value in values.items() if name in names}
                logits.append(session.run(None, feed)[0])
        probabilities = _probabilities(np.concatenate(logits))
        scored = frame.with_columns(pl.Series(name, probabilities[:, index])
                                    for index, name in enumerate(PROBABILITY_COLUMNS))
        destination = output_dir / (source.relative_to(path) if path.is_dir() else source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() == ".csv":
            scored.write_csv(destination)
        else:
            scored.write_parquet(destination)
        outputs.append(destination)

    return outputs
