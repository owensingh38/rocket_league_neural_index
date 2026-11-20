from __future__ import annotations

import json
import os
import numpy as np
import polars as pl
from contextlib import nullcontext
from pathlib import Path
from shared import (
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
    INPUT_DIR as LGPS_INPUT_DIR,
    normalize_model_entities,
)
from .targets import output_count
from .config import DEFAULT_MODEL_CONFIG, OUTPUT_DIR as LGPS_OUTPUT_DIR
from .dataset import SOURCE_HZ

ROOT = Path(__file__).resolve().parents[1]

LGPS_INPUT_PATH = Path(LGPS_INPUT_DIR)
LGPS_OUTPUT_PATH = Path(LGPS_OUTPUT_DIR)

if not LGPS_INPUT_PATH.is_absolute():
    LGPS_INPUT_PATH = ROOT / LGPS_INPUT_PATH

MODEL_PATH = LGPS_OUTPUT_DIR / "lgps_transformer.pt"
OUTPUT_DIR = LGPS_OUTPUT_DIR / "inference"

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


def _model_inputs(frame: pl.DataFrame, names: tuple[str, ...], model_entities=None) -> dict[str, np.ndarray]:
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
    if model_entities is not None:
        selected = normalize_model_entities(model_entities)
        layout = (['ball'] + ['players'] * 6 + ['goals'] * 2
                  + ['big_boosts'] * (6 if big_boosts else 0)
                  + ['small_boosts'] * (28 if small_boosts else 0))
        keep = [index for index, name in enumerate(layout) if name in selected]
        types = types[keep]
        entity_teams = entity_teams[:, keep]

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


def _prepare_if_needed(frame: pl.DataFrame, sample_rate: float = 10.0) -> pl.DataFrame:
    if frame.is_empty():
        return frame

    # Parsed replay tables carry source-rate frame markers. Always run those
    # through the trainer's eligibility, roster-strength, goal, and hinged
    # sampling pipeline even when they also contain model features.
    raw_markers = {'frame_number', 'frame_is_active', 'stint_id'}

    if raw_markers.issubset(frame.columns):
        if 'replay_id' in frame.columns and frame['replay_id'].n_unique() > 1:
            return pl.concat([_prepare_if_needed(group, sample_rate)
                              for group in frame.partition_by('replay_id', maintain_order=True)],
                             how='vertical_relaxed')

        frame = frame.sort(pl.col('frame_number').cast(pl.Int64), maintain_order=True)
        updates = []

        if 'frame_delta' not in frame.columns:
            numbers = frame['frame_number'].cast(pl.Int64).to_numpy()
            delta = np.r_[0.0, np.diff(numbers) / SOURCE_HZ]
            updates.append(pl.Series('frame_delta', delta.astype(np.float32)))

        if 'frame_touch_player_slot' not in frame.columns:
            updates.append(pl.lit(-1, dtype=pl.Int64).alias('frame_touch_player_slot'))

        for pad in range(34):
            name = f'boost_pad_{pad:02d}_pos_z'

            if name not in frame.columns:
                updates.append(pl.lit(70.0, dtype=pl.Float32).alias(name))

        if updates:
            frame = frame.with_columns(updates)

        from .dataset import prepare_replay

        return prepare_replay(frame, inference=True, target_sample_rate=sample_rate)

    context_ready = all(feature in frame.columns for feature in CONTEXT_FEATURES)
    players_ready = all(
        f'player_{slot}_{feature}' in frame.columns
        for slot in range(1, 7) for feature in PLAYER_FEATURES
    )

    if set(BALL_FEATURES).issubset(frame.columns) and context_ready and players_ready:
        updates = []

        for pad in range(34):
            name = f'boost_pad_{pad:02d}_pos_z'

            if name not in frame.columns:
                # Standard Soccar boost pads sit 70 uu above the floor.
                updates.append(pl.lit(70.0).cast(pl.Float32).alias(name))

        return frame.with_columns(updates)

    if "frame_delta" not in frame.columns or "frame_number" not in frame.columns:
        raise ValueError("Expected model-ready LGPS columns or parsed replay frames")

    # Raw parsed replays need the repository's roster mapping and feature
    # preparation. Import it only for those inputs; prepared files stay light.
    from .dataset import prepare_replay

    if "replay_id" not in frame.columns:
        return prepare_replay(frame, inference=True, target_sample_rate=sample_rate)

    return pl.concat([prepare_replay(group, inference=True, target_sample_rate=sample_rate)
                      for group in frame.partition_by("replay_id", maintain_order=True)],
                     how="vertical")


def _probabilities(logits: np.ndarray, targets_per_team: int | None = None,
                   horizon_seconds: float | None = None,
                   two_way_prediction: bool | None = None) -> np.ndarray:
    logits = np.asarray(logits)

    if logits.ndim != 2:
        raise ValueError('LGPS logits must be a 2D array')

    if two_way_prediction is None:
        two_way_prediction = logits.shape[1] % 2 == 0

    if two_way_prediction:
        if logits.shape[1] < 2 or logits.shape[1] % 2 != 0:
            raise ValueError('Two-way LGPS logits must have an even full output width of at least 2')

        per_team = logits.shape[1] // 2
    else:
        if logits.shape[1] < 3 or logits.shape[1] % 2 != 1:
            raise ValueError('LGPS logits must have an odd full output width of at least 3')

        per_team = (logits.shape[1] - 1) // 2

    if targets_per_team is not None:
        expected = output_count(targets_per_team, horizon_seconds, two_way_prediction)

        if logits.shape[1] != expected:
            raise ValueError(f'logit width {logits.shape[1]} does not match model config ({expected})')

    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    values /= values.sum(axis=1, keepdims=True)
    neither = np.zeros(len(values), dtype=values.dtype) if two_way_prediction else values[:, -1]
    orange_end = per_team * 2 if two_way_prediction else -1
    return np.column_stack((values[:, :per_team].sum(axis=1),
                            values[:, per_team:orange_end].sum(axis=1),
                            neither)).astype(np.float32, copy=False)


def load_native_lgps_model(model_path, device):
    from .model import (
        LGPSTransformer, lgps_checkpoint_config, load_lgps_checkpoint, load_lgps_weights,
    )

    checkpoint = load_lgps_checkpoint(model_path, map_location='cpu')
    config, state = lgps_checkpoint_config(checkpoint)
    aux_tasks = checkpoint.get('train_config', {}).get('aux_tasks', ())
    model = LGPSTransformer(config, aux_tasks=aux_tasks).to(device).eval()
    load_lgps_weights(model, state, include_aux=bool(aux_tasks))
    return model


def _native_lgps_inputs(model):
    return tuple(name for name in INPUT_NAMES if name not in ('big_boosts', 'small_boosts')
                 or name in model.model_entities)


def _native_lgps_logits(model, tensors, start, end, precision):
    import torch

    feed = {name: value if name == 'entity_types' else value[start:end]
            for name, value in tensors.items()}
    autocast = torch.autocast('cuda', dtype=torch.float16) if precision == 'fp16' else nullcontext()
    with torch.inference_mode(), autocast:
        return model(**feed).cpu().numpy()


def predict_lgps_probabilities(frame, model, *, batch_size=512):
    """Score prepared replay frames with a loaded native LGPS model."""
    import torch

    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if frame.is_empty():
        return np.empty((0, len(PROBABILITY_COLUMNS)), dtype=np.float32)

    device = next(model.parameters()).device
    values = _model_inputs(frame, _native_lgps_inputs(model), model.model_entities)
    tensors = {name: torch.from_numpy(value).to(device) for name, value in values.items()}
    logits = np.concatenate([_native_lgps_logits(model, tensors, start, start + batch_size, 'fp32')
                             for start in range(0, frame.height, batch_size)])
    config = model.config
    return _probabilities(logits, config.targets_per_team, config.horizon_seconds, config.two_way_prediction)


def apply_lgps(path: str | Path, model_path: str | Path = MODEL_PATH,
               output_dir: str | Path = OUTPUT_DIR, batch_size: int | None = None,
               threads: int | None = None, device: str = "cpu",
               precision: str = "fp32", sample_rate: float | None = None,
               fix_context: bool = False) -> list[Path]:
    """Write scored replay files using the model's cadence.

    ``sample_rate`` supplies the cadence for legacy exports without metadata.
    """
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
    native = model_path.suffix.lower() == '.pt' and not torchscript
    output_metadata = {}
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be fp32 or fp16")

    config = None
    has_context_token = None

    if native:
        import torch

        torch.set_num_threads(threads)

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')

        if device not in ('cpu', 'cuda'):
            raise ValueError('Native LGPS device must be cpu or cuda')

        if precision == 'fp16' and device != 'cuda':
            raise ValueError('fp16 inference requires CUDA')

        model = load_native_lgps_model(model_path, device)
        config = model.config
        output_metadata = {
            'sample_rate': config.sample_rate,
            'targets_per_team': config.targets_per_team,
            'horizon_seconds': config.horizon_seconds,
            'two_way_prediction': config.two_way_prediction,
        }
        entities = set(model.model_entities)
        output_metadata['model_entities'] = model.model_entities
        has_context_token = "context" in entities
        output_metadata["has_context_token"] = has_context_token
        names = _native_lgps_inputs(model)
    elif torchscript:
        import torch

        if precision == "fp16" and device != "cuda":
            raise ValueError("fp16 inference requires a CUDA TorchScript model")

        torch.set_num_threads(threads)

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device not in ("cpu", "cuda"):
            raise ValueError("TorchScript device must be cpu or cuda")

        extra_files = {'lgps_config.json': ''}
        model = torch.jit.load(str(model_path), map_location="cpu",
                               _extra_files=extra_files).to(device).eval()

        if extra_files['lgps_config.json']:
            output_metadata = json.loads(extra_files['lgps_config.json'])

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
        output_metadata = {
            key.removeprefix('lgps.'): json.loads(value)
            for key, value in session.get_modelmeta().custom_metadata_map.items()
            if key.startswith('lgps.')
        }
        names = tuple(value.name for value in session.get_inputs())

    if not set(names).issubset(INPUT_NAMES):
        raise ValueError(f"Unsupported LGPS model inputs: {names}")

    fallback_rate = DEFAULT_MODEL_CONFIG.sample_rate if sample_rate is None else sample_rate
    sample_rate = output_metadata.get('sample_rate', fallback_rate)
    if not np.isfinite(sample_rate) or not 0 < sample_rate <= SOURCE_HZ:
        raise ValueError(f'sample_rate must lie in (0, {SOURCE_HZ:g}]')

    if batch_size is None:
        batch_size = output_metadata.get('batch_size', TORCHSCRIPT_BATCH_SIZE) if torchscript else 512
    if batch_size < 1:
        raise ValueError('batch_size must be positive')

    if has_context_token is None:
        has_context_token = bool(output_metadata.get("has_context_token", "game_context" in names))

    outputs = []

    for source in _input_paths(path, output_dir):
        frame = _prepare_if_needed(_read_frame(source), sample_rate=sample_rate)

        if frame.is_empty():
            continue

        values = _model_inputs(frame, names, output_metadata.get('model_entities'))

        if fix_context and has_context_token and "game_context" in names:
            context = values["game_context"]
            context[:, CONTEXT_FEATURES.index("game_seconds_remaining")] = 300.0
            context[:, CONTEXT_FEATURES.index("is_overtime")] = 0.0
            context[:, CONTEXT_FEATURES.index("blue_score")] = 0.0
            context[:, CONTEXT_FEATURES.index("orange_score")] = 0.0

        logits = []
        tensors = ({name: torch.from_numpy(value).to(device) for name, value in values.items()}
                   if torchscript or native else None)

        for start in range(0, frame.height, batch_size):
            end = min(start + batch_size, frame.height)

            if native:
                logits.append(_native_lgps_logits(model, tensors, start, end, precision))
            elif torchscript:
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

        probabilities = _probabilities(
            np.concatenate(logits),
            output_metadata.get('targets_per_team'),
            output_metadata.get('horizon_seconds'),
            output_metadata.get('two_way_prediction'),
        )
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
