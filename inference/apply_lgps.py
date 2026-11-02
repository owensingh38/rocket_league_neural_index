from __future__ import annotations

import json
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
    INPUT_DIR as LGPS_INPUT_DIR,
)
from lgps.targets import output_count
from lgps.config import LGPSModelConfig, OUTPUT_DIR as LGPS_CONFIG_OUTPUT_DIR

ROOT = Path(__file__).resolve().parents[1]

LGPS_OUTPUT_DIR = (ROOT / "results" / "lgps"
                   if (ROOT / "results" / "lgps" / "lgps_transformer.pt").is_file()
                   else LGPS_CONFIG_OUTPUT_DIR)
LGPS_INPUT_PATH = Path(LGPS_INPUT_DIR)
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


def _prepare_if_needed(frame: pl.DataFrame, player_view: bool = False,
                       sample_rate: float = 10.0) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if '_window_id' in frame.columns and 'stint_id' not in frame.columns:
        frame = frame.rename({'_window_id': 'stint_id'})

    if player_view:
        from lgps.dataset import PLAYER_CAMERA_FIELDS

        has_camera = all(f'player_{slot}_{field}' in frame.columns
                         for slot in range(1, 7)
                         for field in ('id', *PLAYER_CAMERA_FIELDS))
    else:
        has_camera = True
    if (set(BALL_FEATURES).issubset(frame.columns) and set(CONTEXT_FEATURES).issubset(frame.columns)
            and has_camera):
        return frame

    if "frame_delta" not in frame.columns or "frame_number" not in frame.columns:
        if player_view and not has_camera:
            raise ValueError('Camera-enabled inference requires recorded player camera fields; '
                             'regenerate model-ready data with include_player_camera=True')
        raise ValueError("Expected model-ready LGPS columns or parsed replay frames")

    # Raw parsed replays need the repository's roster mapping and feature
    # preparation. Import it only for those inputs; prepared files stay light.
    from lgps.dataset import prepare_replay

    if "replay_id" not in frame.columns:
        return prepare_replay(frame, inference=True, target_sample_rate=sample_rate,
                              include_player_camera=player_view)

    return pl.concat([prepare_replay(group, inference=True, target_sample_rate=sample_rate,
                                     include_player_camera=player_view)
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


def apply_lgps(path: str | Path, model_path: str | Path = MODEL_PATH,
               output_dir: str | Path = OUTPUT_DIR, batch_size: int | None = None,
               threads: int | None = None, device: str = "cpu",
               precision: str = "fp32", sample_rate: float = 10.0) -> list[Path]:
    # Write scored replay files and return their output paths.
    path = Path(path)
    path = path.expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not np.isfinite(sample_rate) or sample_rate <= 0 or sample_rate > 30:
        raise ValueError("sample_rate must lie in (0, 30]")
    sample_step = int(round(30.0 / sample_rate))
    if not np.isclose(sample_step * sample_rate, 30.0):
        raise ValueError("sample_rate must evenly divide the 30 Hz source")
    threads = min(os.cpu_count() or 1, 8) if threads is None else threads
    if threads < 1:
        raise ValueError("threads must be positive")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    torchscript = model_path.name.endswith(".torchscript.pt")
    native = model_path.suffix.lower() == '.pt' and not torchscript
    output_metadata = {}
    batch_size = (TORCHSCRIPT_BATCH_SIZE if torchscript else 512) if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be fp32 or fp16")
    camera_config = None
    config = None
    if native:
        import torch
        from lgps.camera import PlayerCameraViewEncoder
        from lgps.model import (
            LGPSTransformer,
            load_lgps_checkpoint,
            load_lgps_weights,
            lgps_checkpoint_config,
        )

        torch.set_num_threads(threads)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        if device not in ('cpu', 'cuda'):
            raise ValueError('Native LGPS device must be cpu or cuda')
        if precision == 'fp16' and device != 'cuda':
            raise ValueError('fp16 inference requires CUDA')
        checkpoint = load_lgps_checkpoint(model_path, map_location='cpu')
        config, checkpoint_state = lgps_checkpoint_config(checkpoint)
        output_metadata = {
            'targets_per_team': config.targets_per_team,
            'horizon_seconds': config.horizon_seconds,
            'two_way_prediction': config.two_way_prediction,
        }
        train_config = checkpoint.get('train_config', {})
        camera_settings = train_config.get('player_view')
        if camera_settings is True or (camera_settings is None and train_config.get('player_view', False)):
            camera_settings = {
                'history_seconds': train_config.get('player_view_history_seconds', 2.0),
                'sample_rate': train_config.get('player_view_sample_rate', train_config.get('player_view_hz', 10)),
                'temporal_blocks': train_config.get('player_view_temporal_blocks', 2),
                'attention_heads': train_config.get('player_view_attention_heads', 8),
                'feedforward_dim': train_config.get('player_view_feedforward_dim', 256),
                'dropout': train_config.get('player_view_dropout', 0.05),
                'camera_mode': train_config.get('player_view_camera_mode', 'recorded'),
                'fov_degrees': train_config.get('player_view_fov_degrees', 110.0),
                'distance': train_config.get('player_view_distance', 270.0),
                'height': train_config.get('player_view_height', 100.0),
                'angle_degrees': train_config.get('player_view_angle_degrees', -3.0),
                'aspect_ratio': train_config.get('player_view_aspect_ratio', 16.0 / 9.0),
            }
        if camera_settings and not isinstance(camera_settings, PlayerCameraViewEncoder.Settings):
            camera_settings = PlayerCameraViewEncoder.Settings.from_saved(camera_settings)
        camera_config = camera_settings if camera_settings else None
        model = LGPSTransformer(
            config, player_view=bool(camera_config),
            camera_encoder_settings=camera_config,
        ).to(device).eval()
        load_lgps_weights(model, checkpoint_state)
        entities = set(model.model_entities)
        names = tuple(name for name in INPUT_NAMES if name not in ('big_boosts', 'small_boosts')
                      or name in entities)
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

    outputs = []
    for source in _input_paths(path, output_dir):
        frame = _prepare_if_needed(_read_frame(source), player_view=bool(camera_config),
                                   sample_rate=sample_rate)
        if frame.is_empty():
            continue
        values = _model_inputs(frame, names)
        logits = []
        tensors = ({name: torch.from_numpy(value).to(device) for name, value in values.items()}
                   if torchscript or native else None)
        observations = None
        if camera_config:
            from lgps.camera import camera_frame_observations, camera_histories

            camera_options = dict(
                mode=camera_config.camera_mode,
                fov_degrees=camera_config.fov_degrees,
                distance=camera_config.distance,
                height=camera_config.height,
                angle_degrees=camera_config.angle_degrees,
                aspect_ratio=camera_config.aspect_ratio,
                model_entities=model.model_entities,
            )
            observations = camera_frame_observations(frame, **camera_options)
        for start in range(0, frame.height, batch_size):
            end = min(start + batch_size, frame.height)
            if native:
                feed = {name: value if name == 'entity_types' else value[start:end]
                        for name, value in tensors.items() if name in names}
                if camera_config:
                    histories = camera_histories(
                        frame, range(start, end),
                        history_seconds=camera_config.history_seconds,
                        sample_rate=camera_config.sample_rate,
                        model_entities=model.model_entities,
                        observations=observations,
                    )
                    feed.update({name: value.to(device) for name, value in histories.items()})
                autocast = (torch.autocast('cuda', dtype=torch.float16)
                            if precision == 'fp16' else nullcontext())
                with torch.inference_mode(), autocast:
                    logits.append(model(**feed).cpu().numpy())
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
