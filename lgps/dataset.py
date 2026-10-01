from __future__ import annotations
import io
import math
import tarfile
import cloudpickle
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset
from props import (
    BALL_FEATURES,
    BIG_BOOST_COLUMNS,
    BLUE_GOAL,
    BOOST_COLUMNS,
    BOOST_ENTITY_GROUPS,
    BOOST_FEATURES,
    BOOST_POSITIONS,
    CONTEXT_FEATURES,
    GOAL_COLUMNS,
    GOAL_FEATURES,
    GOAL_STATE_COLUMNS,
    MAX_PLAYERS_PER_TEAM,
    ORANGE_GOAL,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
    SMALL_BOOST_COLUMNS,
    build_frame_batch,
    normalize_model_entities,
)

TARGET_SAMPLE_RATE = 10.0
SOURCE_HZ = 30.0
RECORD_BATCH_ROWS = 65_536
WRITE_BATCH_ROWS = 131_072

OUTPUT_COLUMNS = (
    'replay_id', '_window_id', '_source_time', 'time_until_next_goal', *GOAL_STATE_COLUMNS, *CONTEXT_FEATURES, *BALL_FEATURES,
    *(field for slot in PLAYER_SLOTS for field in (f'player_{slot}_team_id', *(f'player_{slot}_{feature}' for feature in PLAYER_FEATURES))),
    *GOAL_COLUMNS, *BOOST_COLUMNS, 'random_augmentation',
)

def split_for_source(source_name: str) -> str:
    parts = str(source_name).replace('\\', '/').lower().split('/')
    return 'test' if any(part == '2026' or part.startswith('2026.') for part in parts) else 'train'


def iter_parquets(root: Path):
    # Open one archive member (or unpacked Parquet) at a time.
    root = Path(root)
    if root.is_file():
        paths = (root,)
    elif root.is_dir():
        paths = tuple(sorted(path for path in root.rglob('*') if path.is_file()
                             and path.suffix.lower() in ('.parquet', '.tar')))
    else:
        raise FileNotFoundError(f'LGPS source does not exist: {root}')
    if not paths:
        raise FileNotFoundError(f'No replay Parquet files or archives under: {root}')
    for path in paths:
        if path.suffix.lower() == '.parquet':
            yield str(path), pq.ParquetFile(path)
        elif path.suffix.lower() == '.tar':
            with tarfile.open(path, 'r:') as bundle:
                for member in bundle:
                    if member.isfile() and member.name.endswith('_frames.parquet'):
                        stream = bundle.extractfile(member)
                        if stream is not None:
                            yield f'{path}/{member.name}', pq.ParquetFile(io.BytesIO(stream.read()))


PLAYER_CORE_FEATURES = (
    'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w',
)


def source_player_slots(available):
    # Return physical source slots, excluding metadata-only ghost slots.
    slots = sorted({
        int(name.split('_')[1]) for name in available
        if name.startswith('player_') and name.endswith('_team_id') and name.split('_')[1].isdigit()
    })
    return tuple(
        slot for slot in slots
        if {f'player_{slot}_{feature}' for feature in ('team_id', 'on_field', 'demoed', *PLAYER_CORE_FEATURES)}.issubset(available)
    )


def selected_source_columns(available):
    # Project causal fields and every usable source-player stream.
    required = {
        'replay_id', 'frame_number', 'frame_delta', 'frame_seconds_elapsed', 'frame_is_active', 'stint_id',
        'blue_team_score', 'orange_team_score', *BALL_FEATURES,
    }
    missing = required.difference(available)
    if missing:
        raise ValueError(f'missing required source columns: {sorted(missing)}')
    slots = source_player_slots(available)
    if not slots:
        raise ValueError('no physical player slots with complete position, velocity, and rotation fields')
    player_columns = {
        f'player_{slot}_{feature}'
        for slot in slots
        for feature in ('team_id', *(feature for feature in PLAYER_FEATURES if feature != 'respawn_timer'))
        if f'player_{slot}_{feature}' in available
    }
    optional = {
        name for name in ('goal_event_id', 'goal_event_team_id', 'blue_next_goal', 'orange_next_goal')
        if name in available
    }
    boosts = {name for name in available if name.startswith('boost_pickup_')}
    return sorted(required | player_columns | optional | boosts)


def iter_replays(parquet, columns):
    # Yield a contiguous replay while keeping bounded Parquet batches in memory.
    pending = None
    completed = set()
    for batch in parquet.iter_batches(batch_size=RECORD_BATCH_ROWS, columns=columns):
        frame = pl.from_arrow(batch)
        for replay_id in frame['replay_id'].unique(maintain_order=True).to_list():
            part = frame.filter(pl.col('replay_id') == replay_id)
            if pending is not None and replay_id == pending['replay_id'][0]:
                pending = pl.concat((pending, part), how='vertical_relaxed')
            else:
                if pending is not None:
                    completed.add(pending['replay_id'][0])
                    yield pending
                if replay_id in completed:
                    raise ValueError(f'replay {replay_id} is not contiguous in its source parquet')
                pending = part
    if pending is not None:
        yield pending


def numeric(frame, column, default=0.0):
    # Return a writable float32 array; replay batches can expose read-only Arrow views.
    if column not in frame.columns:
        return np.full(frame.height, default, dtype=np.float32)
    return frame[column].cast(pl.Float32, strict=False).fill_null(default).to_numpy().astype(np.float32, copy=True)


def goal_rows_and_teams(frame):
    # Use native goal events, or score increments when historical fields are absent.
    count = frame.height
    if {'goal_event_id', 'goal_event_team_id'}.issubset(frame.columns):
        event_ids = frame['goal_event_id'].cast(pl.String, strict=False).fill_null('').to_numpy()
        teams = frame['goal_event_team_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
        rows = (event_ids != '') & np.isin(teams, (0, 1))
        rows[1:] &= (event_ids[1:] != event_ids[:-1]) | ~rows[:-1]
        return rows, teams.astype(np.int8, copy=False)
    blue = numeric(frame, 'blue_team_score', np.nan)
    orange = numeric(frame, 'orange_team_score', np.nan)
    blue_scored = np.r_[False, np.diff(blue) > 0]
    orange_scored = np.r_[False, np.diff(orange) > 0]
    rows = blue_scored ^ orange_scored
    teams = np.where(blue_scored, 0, np.where(orange_scored, 1, -1)).astype(np.int8)
    return rows, teams


def resolve_match_roster(frame):
    # Choose each team’s three sustained physical streams and strength mask.

    # Metadata-only slots are rejected by ``source_player_slots``. A transient
    # physical stream is ignored unless its on-field exposure is material
    # relative to the third-most-present player on its team. This preserves
    # ordinary 3v3 replays with a ghost/seventh slot while exposing real 4v3,
    # 3v4, and 4v4 states for filtering.

    slots = source_player_slots(set(frame.columns))
    by_team = {0: [], 1: []}
    active_by_slot = {}
    for slot in slots:
        prefix = f'player_{slot}_'
        team_values = frame[prefix + 'team_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
        known = team_values[np.isin(team_values, (0, 1))]
        if not known.size:
            continue
        team = int(np.bincount(known, minlength=2).argmax())
        position = np.column_stack([numeric(frame, prefix + axis, np.nan) for axis in ('pos_x', 'pos_y', 'pos_z')])
        active = numeric(frame, prefix + 'on_field', 0.0).astype(bool) & ~numeric(frame, prefix + 'demoed', 0.0).astype(bool)
        exposure = int((active & np.isfinite(position).all(axis=1)).sum())
        by_team[team].append((slot, exposure))
        active_by_slot[slot] = active
    if any(len(by_team[team]) < 3 for team in (0, 1)):
        raise ValueError('could not identify three physical source players for each team')
    selected = {}
    strength_slots = {0: (), 1: ()}
    for team in (0, 1):
        ranked = sorted(by_team[team], key=lambda item: (-item[1], item[0]))
        third_exposure = ranked[2][1]
        if third_exposure <= 0:
            raise ValueError(f'team {team} has fewer than three observed on-field players')
        selected[team] = tuple(slot for slot, _ in ranked[:3])
        # A fourth genuine participant has substantial exposure; one-frame
        # parser ghosts do not. This set is only used for strength filtering.
        relevance_floor = max(1, int(np.ceil(third_exposure * 0.25)))
        strength_slots[team] = tuple(slot for slot, exposure in ranked if exposure >= relevance_floor)
    blue_strength = sum(active_by_slot[slot].astype(np.int8) for slot in strength_slots[0])
    orange_strength = sum(active_by_slot[slot].astype(np.int8) for slot in strength_slots[1])
    invalid = ((blue_strength == 4) & (orange_strength == 3)) | ((blue_strength == 3) & (orange_strength == 4)) | ((blue_strength == 4) & (orange_strength == 4))
    return selected, ~invalid


def remap_match_roster(frame, selected):
    # Map dynamic source slots to the model’s stable blue-then-orange six slots.
    mapping = {source: destination for destination, source in enumerate((*selected[0], *selected[1]), start=1)}
    updates = []
    for source, destination in mapping.items():
        source_prefix, target_prefix = f'player_{source}_', f'player_{destination}_'
        team = 0 if destination <= 3 else 1
        updates.append(pl.lit(team, dtype=pl.Int64).alias(target_prefix + 'team_id'))
        for feature in PLAYER_FEATURES:
            source_column = source_prefix + feature
            if source_column in frame.columns:
                updates.append(pl.col(source_column).alias(target_prefix + feature))
            else:
                updates.append(pl.lit(0.0, dtype=pl.Float32).alias(target_prefix + feature))
    for column in (name for name in frame.columns if name.startswith('boost_pickup_event_player_') and name.endswith('_slot')):
        values = frame[column].to_list()
        updates.append(pl.Series(column, [mapping.get(int(value)) if value is not None and int(value) in mapping else None for value in values], dtype=pl.Int64))
    return frame.with_columns(updates)


def attach_boost_state(frame):
    # Causally reconstruct pad availability without iterating every frame.
    times = frame['_source_time'].to_numpy()
    stints = frame['stint_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    values = np.zeros((frame.height, 34, len(BOOST_FEATURES)), dtype=np.float32)
    values[..., :2] = BOOST_POSITIONS
    values[:, :6, 2] = 1.0
    values[:, 6:, 3] = 1.0
    cooldown = np.r_[np.full(6, 10.0), np.full(28, 5.0)]
    pickup_slots = [name for name in frame.columns if name.startswith('boost_pickup_event_player_') and name.endswith('_slot')]
    event_times = [[] for _ in range(34)]
    ready_times = [[] for _ in range(34)]
    stint_starts = np.flatnonzero(np.r_[True, stints[1:] != stints[:-1]])
    for index in stint_starts:
        for pad in range(34):
            event_times[pad].append(times[index])
            ready_times[pad].append(times[index])
    if {'boost_pickup_event_id', 'boost_pickup_type'}.issubset(frame.columns):
        event_ids = frame['boost_pickup_event_id'].to_list()
        kinds = frame['boost_pickup_type'].to_list()
        slots = {column: frame[column].to_list() for column in pickup_slots}
        positions = {
            slot: np.column_stack([numeric(frame, f'player_{slot}_pos_{axis}', np.nan) for axis in 'xyz'])
            for slot in PLAYER_SLOTS
        }
        seen = set()
        for index, event in enumerate(event_ids):
            if event is None or event in seen:
                continue
            seen.add(event)
            kind = kinds[index]
            if kind == 'reset':
                for pad in range(34):
                    event_times[pad].append(times[index])
                    ready_times[pad].append(times[index])
                continue
            if kind not in ('big', 'small'):
                continue
            candidates = np.arange(6) if kind == 'big' else np.arange(6, 34)
            for column in pickup_slots:
                slot = slots[column][index]
                if slot is None or int(slot) not in PLAYER_SLOTS:
                    continue
                position = positions[int(slot)][index]
                if not np.isfinite(position).all() or position[2] >= 250:
                    continue
                distance = np.linalg.norm(BOOST_POSITIONS[candidates] - position[:2], axis=1)
                if distance.min() <= 450:
                    pad = int(candidates[distance.argmin()])
                    event_times[pad].append(times[index])
                    ready_times[pad].append(times[index] + cooldown[pad])
    for pad in range(34):
        event = np.asarray(event_times[pad], dtype=np.float64)
        ready = np.asarray(ready_times[pad], dtype=np.float64)
        order = np.argsort(event, kind='stable')
        event, ready = event[order], ready[order]
        prior = np.searchsorted(event, times, side='right') - 1
        refill = np.zeros(frame.height, dtype=np.float32)
        valid = prior >= 0
        refill[valid] = np.maximum(ready[prior[valid]] - times[valid], 0.0)
        values[:, pad, 5] = refill
        values[:, pad, 4] = refill <= 0.0
    return frame.with_columns(pl.Series(name, values.reshape(frame.height, -1)[:, index]) for index, name in enumerate(BOOST_COLUMNS))


def apply_random_augmentation(frame):
    # Apply a selected 180-degree field rotation to every world-state field.
    selected = frame['random_augmentation'].to_numpy().astype(bool, copy=False)
    if not selected.any():
        return frame
    updates = []
    for prefix in ('ball', *(f'player_{slot}' for slot in PLAYER_SLOTS)):
        # Position, linear velocity, and angular velocity are world vectors.
        for field in ('pos', 'vel', 'ang_vel'):
            for axis in ('x', 'y'):
                column = f'{prefix}_{field}_{axis}'
                values = numeric(frame, column, np.nan)
                values[selected] *= -1.0
                updates.append(pl.Series(column, values))
        quaternion_columns = tuple(f'{prefix}_rot_{axis}' for axis in 'xyzw')
        quaternion = np.column_stack([numeric(frame, column, np.nan) for column in quaternion_columns])
        rotated = quaternion[selected]
        x, y, z, w = rotated.T
        rotated = np.column_stack((-y, x, w, -z))
        norm = np.linalg.norm(rotated, axis=1, keepdims=True)
        rotated = np.divide(rotated, norm, out=np.full_like(rotated, np.nan), where=norm > 1e-8)
        rotated *= np.where(rotated[:, 3:4] < 0.0, -1.0, 1.0)
        quaternion[selected] = rotated
        updates.extend(pl.Series(column, quaternion[:, index]) for index, column in enumerate(quaternion_columns))
        if prefix.startswith('player_'):
            x, y, z, w = quaternion.T
            angles = {
                'pitch': np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
                'yaw': np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
                'roll': np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            }
            for field, angle in angles.items():
                values = numeric(frame, prefix + field, np.nan)
                values[selected] = angle[selected]
                updates.append(pl.Series(prefix + field, values.astype(np.float32)))
    for prefix in ('blue_goal', 'orange_goal'):
        for field in ('pos', 'opening_direction'):
            for axis in ('x', 'y'):
                column = f'{prefix}_{field}_{axis}'
                values = numeric(frame, column, np.nan)
                values[selected] *= -1.0
                updates.append(pl.Series(column, values))
    for pad in range(34):
        for axis in ('x', 'y'):
            column = f'boost_pad_{pad:02d}_pos_{axis}'
            values = numeric(frame, column, np.nan)
            values[selected] *= -1.0
            updates.append(pl.Series(column, values))
    return frame.with_columns(updates)


def prepare_replay(frame, random_augmentation=False):
    # Compile one replay with exact targets, dynamic roster mapping, and valid strength states.
    if TARGET_SAMPLE_RATE <= 0 or TARGET_SAMPLE_RATE > SOURCE_HZ:
        raise ValueError('TARGET_SAMPLE_RATE must lie in (0, 30]')
    step = int(round(SOURCE_HZ / TARGET_SAMPLE_RATE))
    if not np.isclose(step * TARGET_SAMPLE_RATE, SOURCE_HZ):
        raise ValueError('TARGET_SAMPLE_RATE must evenly divide the 30 Hz source')
    frame = frame.sort('frame_number')
    frame_numbers = frame['frame_number'].cast(pl.Int64).to_numpy()
    if np.any(np.diff(frame_numbers) <= 0):
        raise ValueError('frame_number must be strictly increasing within a replay')
    selected, valid_strength = resolve_match_roster(frame)
    frame = remap_match_roster(frame, selected)
    source_time = np.cumsum(numeric(frame, 'frame_delta', 0.0), dtype=np.float64)
    goal_row, teams = goal_rows_and_teams(frame)
    stints = frame['stint_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    goal_time = np.full(frame.height, np.nan, dtype=np.float64)
    goal_team = np.full(frame.height, -1, dtype=np.int8)
    scoring_stints = set()
    for stint in np.unique(stints):
        rows = np.flatnonzero(stints == stint)
        goals = rows[goal_row[rows]]
        if goals.size:
            goal_time[rows] = source_time[goals[0]]
            goal_team[rows] = teams[goals[0]]
            scoring_stints.add(stint)
    frame = frame.with_columns([
        pl.Series('_source_time', source_time), pl.Series('_window_id', stints),
        pl.Series('time_until_next_goal', goal_time - source_time),
        pl.Series('_goal_row', goal_row),
    ])
    time_until_goal = goal_time - source_time
    within_ten_seconds = np.isfinite(time_until_goal) & (time_until_goal > 0.0) & (time_until_goal <= 10.0)
    blue_target = within_ten_seconds & (goal_team == 0)
    orange_target = within_ten_seconds & (goal_team == 1)
    frame = frame.with_columns([
        pl.Series('blue_goal_in_next_ten_seconds', blue_target),
        pl.Series('orange_goal_in_next_ten_seconds', orange_target),
        pl.Series('neither_goal_in_next_ten_seconds', ~(blue_target | orange_target)),
    ])
    pre_goal = np.r_[goal_row[1:], False]
    keep = ~goal_row
    active = frame['frame_is_active'].cast(pl.Boolean, strict=False).fill_null(False).to_numpy()
    keep &= active & valid_strength
    frame = frame.filter(pl.Series('_keep_active_strength', keep))
    anchors = pre_goal[keep]
    kept_stints = frame['stint_id'].cast(pl.Int64).fill_null(-1).to_numpy()
    sample = np.zeros(frame.height, dtype=bool)
    for stint in np.unique(kept_stints):
        rows = np.flatnonzero(kept_stints == stint)
        anchor_rows = rows[anchors[rows]]
        if anchor_rows.size:
            sample[np.arange(anchor_rows[-1], rows[0] - 1, -step)] = True
    frame = frame.filter(pl.Series('_sample', sample))
    frame = frame.filter(pl.col('stint_id').cast(pl.Int64).is_in(list(scoring_stints)))
    frame = frame.filter((pl.col('blue_team_score').cast(pl.Float32) - pl.col('orange_team_score').cast(pl.Float32)).abs() < 4.0)
    if frame.is_empty():
        return frame
    current_stints = frame['stint_id'].cast(pl.Int64).fill_null(-1).to_numpy()
    updates = [
        pl.col('frame_seconds_elapsed').cast(pl.Float32).alias('seconds_elapsed'),
        (pl.col('frame_seconds_elapsed').cast(pl.Float32) >= 300.0).cast(pl.Float32).alias('overtime'),
        *(pl.lit(value, dtype=pl.Float32).alias(f'blue_goal_{feature}')
          for feature, value in zip(GOAL_FEATURES, BLUE_GOAL, strict=True)),
        *(pl.lit(value, dtype=pl.Float32).alias(f'orange_goal_{feature}')
          for feature, value in zip(GOAL_FEATURES, ORANGE_GOAL, strict=True)),
        pl.Series(
            'random_augmentation',
            np.random.random(frame.height) < 0.5 if random_augmentation else np.zeros(frame.height, dtype=bool),
        ),
    ]
    for slot in PLAYER_SLOTS:
        prefix = f'player_{slot}_'
        demoed = numeric(frame, prefix + 'demoed', 0.0).astype(bool)
        on_field = numeric(frame, prefix + 'on_field', 0.0).astype(bool)
        now = frame['_source_time'].to_numpy()
        starts = demoed & ~np.r_[False, demoed[:-1]]
        starts[1:] |= demoed[1:] & (current_stints[1:] != current_stints[:-1])
        last_start = np.maximum.accumulate(np.where(starts, now, -np.inf))
        updates.extend([
            pl.Series(prefix + 'respawn_timer', np.where(demoed, np.clip(3.0 - (now - last_start), 0, 3), 0).astype(np.float32)),
            pl.Series(prefix + 'demoed', demoed.astype(np.float32)),
            pl.Series(prefix + 'on_field', (on_field & ~demoed).astype(np.float32)),
        ])
        for feature in PLAYER_FEATURES:
            if feature not in ('respawn_timer', 'demoed', 'on_field'):
                updates.append(
                    pl.when(pl.col(prefix + 'demoed').cast(pl.Boolean, strict=False).fill_null(False))
                    .then(float('nan'))
                    .otherwise(pl.col(prefix + feature).cast(pl.Float32, strict=False).fill_null(0.0))
                    .alias(prefix + feature)
                )
    frame = attach_boost_state(frame.with_columns(updates))
    frame = apply_random_augmentation(frame)
    return frame.select(OUTPUT_COLUMNS)

class LGPSScaler:
    # AnalyzeRL-Boxcars uses Rocket League world units, scaled boost in [0, 100],
    # quaternion components in [-1, 1], and respawn timers in seconds.
    BALL_SCALE = torch.tensor(
        (
            5000.0, 5000.0, 5000.0,
            2500.0, 2500.0, 2500.0,
            360.0, 360.0, 360.0,
            1.0, 1.0, 1.0, 1.0,
        ),
        dtype=torch.float32,
    )
    PLAYER_SCALE = torch.tensor(
        (
            5000.0, 5000.0, 5000.0,
            2500.0, 2500.0, 2500.0,
            360.0, 360.0, 360.0,
            1.0, 1.0, 1.0, 1.0,
            100.0, 3.0,
            1.0, 1.0,
            1.0, 1.0, 1.0,
            math.pi, math.pi, math.pi,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ),
        dtype=torch.float32,
    )
    GOAL_SCALE = torch.tensor(
        (
            5000.0, 5000.0, 5000.0,
            1785.5, 880.0, 642.775,
            1.0, 1.0, 1.0,
        ),
        dtype=torch.float32,
    )
    GEOMETRY_SCALE = torch.tensor(
        (
            5000.0, 5000.0, 5000.0,
            2500.0, 2500.0, 2500.0,
            5000.0, 2500.0, 2500.0, 2500.0, 2500.0,
            1.0, 1.0, 1.0, 1.0,
        ),
        dtype=torch.float32,
    )

    CONTEXT_SCALE = torch.tensor(
        (300.0, 1.0, 10.0, 10.0),
        dtype=torch.float32,
    )

    @staticmethod
    def _scale(values, scale):
        return values.float() / scale.to(
            device=values.device,
            dtype=values.dtype,
        )

    @classmethod
    def ball(cls, values):
        return cls._scale(values, cls.BALL_SCALE)

    @classmethod
    def player(cls, values):
        return cls._scale(values, cls.PLAYER_SCALE)

    @classmethod
    def goal(cls, values):
        return cls._scale(values, cls.GOAL_SCALE)

    @classmethod
    def context(cls, values):
        return cls._scale(values, cls.CONTEXT_SCALE)

    @classmethod
    def geometry(cls, values):
        return cls._scale(values, cls.GEOMETRY_SCALE)

def _numeric(frame: pl.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full(frame.height, default, dtype=np.float32)
    values = frame.get_column(column)
    if values.null_count():
        if math.isnan(default):
            values = values.cast(pl.Float32, strict=False)
        values = values.fill_null(default)
    return values.to_numpy().astype(np.float32, copy=False)

def player_feature_columns(slots):
    return tuple(f"player_{slot}_{feature}" for slot in slots if slot is not None for feature in PLAYER_FEATURES)

def player_team_columns(slots):
    return tuple(f"player_{slot}_team_id" for slot in slots if slot is not None)

def player_slots(frame: pl.DataFrame, max_players_per_team: int) -> tuple[int | None, ...]:
    teams: dict[int, list[int]] = {0: [], 1: []}
    for column in frame.columns:
        if not (column.startswith("player_") and column.endswith("_team_id")):
            continue
        values = frame[column].drop_nulls().unique().to_list()
        if not values:
            continue
        if len(values) != 1:
            raise ValueError(f"{column}: team assignment is not stable")
        team = int(values[0])
        if team not in teams:
            raise ValueError(f"{column}: expected blue/orange team id, got {team}")
        teams[team].append(int(column.split("_")[1]))
    if not teams[0] or not teams[1]:
        raise ValueError("Expected at least one roster player on each team")
    if any(len(slots) > max_players_per_team for slots in teams.values()):
        raise ValueError(f"Expected no more than {max_players_per_team} players per team")
    return tuple(
        slot
        for team in (0, 1)
        for slot in sorted(teams[team])
        + [None] * (max_players_per_team - len(teams[team]))
    )

class EntityTokenizer:
    # Convert prepared replay rows into the configured separate entity streams.

    entity_types = torch.tensor([0, 1, 1, 1, 1, 1, 1, 2, 2] + [3] * 34, dtype=torch.long)
    entity_teams = torch.tensor([2, 0, 0, 0, 1, 1, 1, 0, 1] + [2] * 34, dtype=torch.long)

    def __init__(
        self,
        include_context: bool = True,
        max_players_per_team: int = MAX_PLAYERS_PER_TEAM,
        model_entities="all",
    ) -> None:
        if max_players_per_team != 3:
            raise ValueError("LGPS requires three player slots per team")
        self.include_context = bool(include_context)
        self.max_players_per_team = max_players_per_team
        self.model_entities = normalize_model_entities(model_entities)
        self.entity_types = torch.tensor(
            ([0] if "ball" in self.model_entities else [])
            + ([1] * 6 if "players" in self.model_entities else [])
            + ([2] * 2 if "goals" in self.model_entities else [])
            + [3] * sum(count for name, count, _, _ in BOOST_ENTITY_GROUPS if name in self.model_entities),
            dtype=torch.long,
        )

    def __call__(self, frame: pl.DataFrame) -> dict[str, torch.Tensor]:
        if frame.is_empty():
            raise ValueError("cannot tokenize an empty frame batch")
        # Prepared rows contain six player tokens; their teams can vary by
        # frame in valid 4v2/2v4 states and after a seventh-player substitution.
        has_goals = set(GOAL_COLUMNS).issubset(frame.columns)
        if any(column in frame.columns for column in GOAL_COLUMNS) and not has_goals:
            raise ValueError("all goal geometry columns must be supplied together")
        slots = tuple(range(1, 7)) if has_goals else player_slots(frame, self.max_players_per_team)
        required = {
            *CONTEXT_FEATURES,
            *player_feature_columns(slots),
            *player_team_columns(slots),
        }
        if "ball" in self.model_entities:
            required.update(BALL_FEATURES)
        for name, _, columns, _ in BOOST_ENTITY_GROUPS:
            if name in self.model_entities:
                required.update(columns)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"missing model input columns: {sorted(missing)}")
        ball = np.column_stack([_numeric(frame, column) for column in BALL_FEATURES])
        players = np.stack([
            np.column_stack([
                _numeric(frame, f"player_{slot}_{feature}")
                if slot is not None else np.zeros(frame.height, dtype=np.float32)
                for feature in PLAYER_FEATURES
            ])
            for slot in slots
        ], axis=1)
        context = np.column_stack([_numeric(frame, column) for column in CONTEXT_FEATURES])
        if not self.include_context:
            context.fill(0.0)
        goals = (
            np.column_stack([_numeric(frame, column) for column in GOAL_COLUMNS])
            .reshape(frame.height, 2, len(GOAL_FEATURES))
            if has_goals else np.broadcast_to(
                np.stack((BLUE_GOAL, ORANGE_GOAL)),
                (frame.height, 2, len(GOAL_FEATURES)),
            ).copy()
        )
        if "_rotate_180" in frame.columns and not has_goals:
            rotate = frame["_rotate_180"].to_numpy().astype(bool)
            signs = np.where(
                rotate[:, None, None],
                np.asarray([-1.0, -1.0, 1.0], dtype=np.float32),
                np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            )
            goals[:, :, :3] *= signs
            goals[:, :, 6:9] *= signs
        player_teams = np.column_stack([
            _numeric(frame, f"player_{slot}_team_id", -1)
            if slot is not None else np.full(frame.height, -1, dtype=np.float32)
            for slot in slots
        ]).astype(np.int64)
        player_teams = np.where((player_teams == 0) | (player_teams == 1), player_teams, 2)
        team_parts = []
        if "ball" in self.model_entities:
            team_parts.append(np.full((frame.height, 1), 2, dtype=np.int64))
        if "players" in self.model_entities:
            team_parts.append(player_teams)
        if "goals" in self.model_entities:
            team_parts.extend((np.zeros((frame.height, 1), dtype=np.int64),
                               np.ones((frame.height, 1), dtype=np.int64)))
        for name, count, _, _ in BOOST_ENTITY_GROUPS:
            if name in self.model_entities:
                team_parts.append(np.full((frame.height, count), 2, dtype=np.int64))
        entity_teams = np.concatenate(team_parts, axis=1)
        result = {
            "ball": torch.from_numpy(ball.astype(np.float32, copy=False)),
            "players": torch.from_numpy(players.astype(np.float32, copy=False)),
            "goals": torch.from_numpy(goals),
            "game_context": torch.from_numpy(context.astype(np.float32, copy=False)),
            "entity_types": self.entity_types,
            "entity_teams": torch.from_numpy(entity_teams),
        }
        for name, count, columns, _ in BOOST_ENTITY_GROUPS:
            if name in self.model_entities:
                result[name] = torch.from_numpy(frame.select(columns).to_numpy().astype(np.float32).reshape(frame.height, count, len(BOOST_FEATURES)))
        return result

class LGPSAgents:
    entity_types = EntityTokenizer.entity_types
    entity_teams = EntityTokenizer.entity_teams

    def __init__(self, model_entities="all", *, targets_per_team=10, geometric_width=1.25):
        self.model_entities = normalize_model_entities(model_entities)
        self.tokenizer = EntityTokenizer(model_entities=self.model_entities)
        self.targets_per_team, self.geometric_width = targets_per_team, geometric_width

    def build(self, frame, indices=None):
        return build_frame_batch(frame, range(frame.height) if indices is None else indices,
                             self.tokenizer,
                             targets_per_team=self.targets_per_team,
                             geometric_width=self.geometric_width,
                             )

class LGPSDataSource:
    def __init__(self, path, model_entities="all"):
        self.model_entities = normalize_model_entities(model_entities)
        path = Path(path)
        if path.is_file():
            candidate_paths = (
                (path,)
                if path.suffix.lower() == ".parquet"
                else ()
            )
        elif path.is_dir():
            candidate_paths = tuple(sorted(candidate for candidate in path.rglob("*")
                                           if candidate.is_file() and candidate.suffix.lower() == ".parquet"))
        else:
            raise FileNotFoundError(f"LGPS input path does not exist: {path}")
        if not candidate_paths:
            raise FileNotFoundError(f"No parquet files found under LGPS input path: {path}")

        self.paths = candidate_paths
        self.read_columns = {
            candidate: self.columns_for_path(candidate)
            for candidate in self.paths
        }
        self.row_groups = {
            candidate: self.row_groups_for_path(candidate)
            for candidate in self.paths
        }
        self.row_counts = {
            candidate: sum(row_group for _, row_group in self.row_groups[candidate])
            for candidate in self.paths
        }
        self.total_rows = sum(self.row_counts.values())

    @staticmethod
    def row_groups_for_path(path):
        # Return row-group starts and lengths without scanning Parquet data.
        metadata = pq.ParquetFile(path).metadata
        start = 0
        groups = []
        for index in range(metadata.num_row_groups):
            length = metadata.row_group(index).num_rows
            groups.append((start, length))
            start += length
        return tuple(groups)

    def columns_for_path(self, path):
        schema = set(pl.read_parquet_schema(path))
        team_columns = tuple(
            column
            for column in sorted(schema)
            if column.startswith("player_")
            and column.endswith("_team_id")
        )
        if not team_columns:
            raise ValueError(f"{path}: no player team columns found")
        slots = tuple(int(column.split("_")[1]) for column in team_columns)
        player_columns = player_feature_columns(slots)
        required = {
            "replay_id", "_source_time", "_window_id", "time_until_next_goal", *GOAL_STATE_COLUMNS,
            *(BIG_BOOST_COLUMNS if "big_boosts" in self.model_entities else ()),
            *(SMALL_BOOST_COLUMNS if "small_boosts" in self.model_entities else ()),
            *CONTEXT_FEATURES,
            *BALL_FEATURES,
            *team_columns,
            *player_columns,
        }
        goal_columns = tuple(column for column in GOAL_COLUMNS if column in schema)
        missing = required.difference(schema)
        if missing:
            raise ValueError(
                f"{path}: missing model-ready columns: {sorted(missing)}"
            )
        return tuple(dict.fromkeys(
            (
            "replay_id", "_source_time", "_window_id", "time_until_next_goal", *GOAL_STATE_COLUMNS,
                *goal_columns,
                *(BIG_BOOST_COLUMNS if "big_boosts" in self.model_entities else ()),
                *(SMALL_BOOST_COLUMNS if "small_boosts" in self.model_entities else ()),
                    *CONTEXT_FEATURES,
                *BALL_FEATURES,
                *team_columns,
                *player_columns,
            )
        ))

    def read_batches(self, path, batch_size=2048, offset=0, length=None):
        # Yield a contiguous range by reading only intersecting row groups.
        end = self.row_counts[path] if length is None else min(
            self.row_counts[path], offset + length
        )
        if offset < 0 or end < offset:
            raise ValueError("invalid parquet row range")
        reader = pq.ParquetFile(path)
        pending = None
        for group_index, (group_start, group_rows) in enumerate(self.row_groups[path]):
            group_end = group_start + group_rows
            if group_end <= offset or group_start >= end:
                continue
            table = reader.read_row_group(
                group_index,
                columns=list(self.read_columns[path]),
            )
            first = max(offset - group_start, 0)
            last = min(end - group_start, group_rows)
            current = pl.from_arrow(table.slice(first, last - first))
            pending = current if pending is None else pl.concat([pending, current])
            while pending.height >= batch_size:
                yield pending.slice(0, batch_size)
                pending = pending.slice(batch_size)
        if pending is not None and pending.height:
            yield pending

class LGPSDataset(IterableDataset):
    def __init__(
        self,
        source,
        builder,
        batch_size,
        rank=0,
        world_size=1,
        split_start=0,
        split_end=None,
        drop_last=False,
    ):
        super().__init__()
        self.drop_last = drop_last
        self.source = source
        self.builder = builder
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.split_start = split_start
        self.split_end = source.total_rows if split_end is None else split_end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1
        shard_rank = self.rank * worker_count + worker_id
        shard_count = self.world_size * worker_count
        split_rows = self.split_end - self.split_start
        if self.drop_last:
            batches_per_shard = (split_rows // self.batch_size) // shard_count
            shard_start = self.split_start + shard_rank * batches_per_shard * self.batch_size
            shard_end = shard_start + batches_per_shard * self.batch_size
        else:
            shard_start = self.split_start + (split_rows * shard_rank) // shard_count
            shard_end = self.split_start + (split_rows * (shard_rank + 1)) // shard_count
        global_row = 0
        for path in self.source.paths:
            path_rows = self.source.row_counts[path]
            path_start, path_end = global_row, global_row + path_rows
            global_row = path_end
            if path_end <= shard_start or path_start >= shard_end:
                continue
            local_start = max(shard_start - path_start, 0)
            local_end = min(shard_end - path_start, path_rows)
            for frame in self.source.read_batches(
                path, batch_size=self.batch_size, offset=local_start,
                length=local_end - local_start,
            ):
                if self.drop_last and frame.height != self.batch_size:
                    break
                yield self.builder.build(frame, range(frame.height))



class SerializedDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)

    def __reduce__(self):
        return cloudpickle.loads, (cloudpickle.dumps(self.dataset),)


class LGPSData:
    # Data pipeline assembled from training, model, and runtime contracts.
    def __init__(self, train_config, model_config, runtime_config, rank=0, world_size=1):
        self.train_config = train_config
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.rank = rank
        self.world_size = world_size
        self.source = LGPSDataSource(train_config.input_path, model_config.model_entities)
        self.builder = LGPSAgents(
            model_config.model_entities,
            targets_per_team=model_config.targets_per_team,
            geometric_width=model_config.geometric_width,
        )

        train_end = int(self.source.total_rows * train_config.train_split)
        validation_end = int(
            self.source.total_rows * (train_config.train_split + train_config.val_split)
        )
        self.ranges = {
            "train": (0, train_end),
            "validation": (train_end, validation_end),
            "test": (validation_end, self.source.total_rows),
        }

    def loader(self, batch_size, split):
        split_start, split_end = self.ranges[split]
        if split_start >= split_end:
            return None

        num_workers = self.runtime_config.data_workers
        if num_workers < 0:
            raise ValueError("data_workers must be non-negative")
        kwargs = {
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": self.runtime_config.pin_memory and torch.cuda.is_available(),
        }
        if num_workers > 0:
            kwargs.update(
                multiprocessing_context="spawn",
                persistent_workers=self.runtime_config.persistent_workers,
                prefetch_factor=self.runtime_config.prefetch_factor,
            )
        dataset = LGPSDataset(
            self.source,
            self.builder,
            batch_size,
            rank=self.rank,
            world_size=self.world_size,
            split_start=split_start,
            split_end=split_end,
            drop_last=(split == "train" and self.runtime_config.drop_last),
        )
        return DataLoader(
            SerializedDataset(dataset) if num_workers else dataset,
            **kwargs,
        )

    def train_loader(self):
        return self.loader(self.train_config.batch_size, "train")

    def validation_loader(self):
        return self.loader(self.train_config.validation_batch_size, "validation")

    def test_loader(self):
        return self.loader(self.train_config.validation_batch_size, "test")


def prepare_dataset(source, output_dir, *, max_replays=None):
    # Accept a Parquet file, tar archive, or nested source directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    seen_replays = set()
    try:
        for source_name, parquet in iter_parquets(Path(source)):
            split = split_for_source(source_name)
            columns = selected_source_columns(set(parquet.schema_arrow.names))
            for replay in iter_replays(parquet, columns):
                replay_id = replay['replay_id'][0]
                if replay_id in seen_replays:
                    raise ValueError(f'duplicate replay_id across source files: {replay_id}')
                seen_replays.add(replay_id)
                prepared = prepare_replay(replay, random_augmentation=(split == 'train'))
                if not prepared.is_empty():
                    if split not in writers:
                        writers[split] = pq.ParquetWriter(
                            output_dir / f'{split}.parquet', prepared.to_arrow().schema,
                            compression='zstd', use_dictionary=False,
                        )
                    writers[split].write_table(prepared.rechunk().to_arrow(), row_group_size=WRITE_BATCH_ROWS)
                if max_replays is not None and len(seen_replays) >= max_replays:
                    break
            if max_replays is not None and len(seen_replays) >= max_replays:
                break
    finally:
        for writer in writers.values():
            writer.close()
    return {split: output_dir / f'{split}.parquet' for split in writers}

