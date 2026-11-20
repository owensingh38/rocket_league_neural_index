from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import tarfile
import tempfile
import time
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset
from shared import (
    BALL_FEATURES,
    BIG_BOOST_COLUMNS,
    BLUE_GOAL,
    BOOST_COLUMNS,
    BOOST_ENTITY_GROUPS,
    BOOST_FEATURES,
    CONTEXT_FEATURES,
    GOAL_COLUMNS,
    GOAL_FEATURES,
    MAX_PLAYERS_PER_TEAM,
    ORANGE_GOAL,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
    SMALL_BOOST_COLUMNS,
    normalize_model_entities,
)
from .targets import build_frame_batch
from .aux_tasks import (
    LGPSBallStateAux, build_aux_targets, configured_aux_tasks,
)

os.environ.setdefault('POLARS_MAX_THREADS', '1')

GOAL_LABEL_COLUMNS = ('blue_goal_next', 'orange_goal_next')
BOOST_POSITIONS = np.asarray([
    (-3072,-4096),(3072,-4096),(-3584,0),(3584,0),(-3072,4096),(3072,4096),
    (-1792,-4184),(0,-4240),(1792,-4184),(-940,-3308),(940,-3308),(0,-2816),
    (-3584,-2484),(-1788,-2300),(1788,-2300),(3584,-2484),(-2048,-1036),(0,-1024),(2048,-1036),
    (-1024,0),(1024,0),(-2048,1036),(0,1024),(2048,1036),(-3584,2484),(-1788,2300),(1788,2300),
    (3584,2484),(0,2816),(-940,3308),(940,3308),(-1792,4184),(0,4240),(1792,4184),
], dtype=np.float32)
BOOST_PAD_Z = np.r_[np.full(6, 73.0), np.full(28, 70.0)].astype(np.float32)

TARGET_SAMPLE_RATE = 10.0
SOURCE_HZ = 30.0
RECORD_BATCH_ROWS = 65_536
WRITE_BATCH_ROWS = 1_048_576
PREPARED_CACHE_VERSION = 15
EVENT_FLAG_COLUMNS = ('kickoff_in_progress', 'goal_in_progress')

OUTPUT_COLUMNS = (
    'replay_id',
    'frame_number',
    'stint_id',
    '_source_time',
    '_stint_ends_with_goal',
    'time_until_next_goal',
    *GOAL_LABEL_COLUMNS,
    'seconds_elapsed',
    *CONTEXT_FEATURES,
    *BALL_FEATURES,
    *(
        field
        for slot in PLAYER_SLOTS
        for field in (
            f'player_{slot}_team_id',
            f'player_{slot}_id',
            *(f'player_{slot}_{feature}' for feature in PLAYER_FEATURES),
        )
    ),
    *GOAL_COLUMNS,
    *BOOST_COLUMNS,
    *(name for name in EVENT_FLAG_COLUMNS if name not in CONTEXT_FEATURES),
    'random_augmentation',
)

PLAYER_CORE_FEATURES = (
    'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w',
)

def iter_parquets(root: Path):
    # Iterate through Parquet files and archive members one at a time.
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
            try:
                yield str(path), pq.ParquetFile(path)
            except (OSError, ValueError, pa.ArrowException) as exc:
                print(f'Skipping unreadable replay-frame Parquet {path}: {exc}', flush=True)

        elif path.suffix.lower() == '.tar':
            try:
                with tarfile.open(path, 'r:') as bundle:
                    for member in bundle:
                        if member.isfile() and member.name.endswith('_frames.parquet'):
                            stream = bundle.extractfile(member)

                            if stream is not None:
                                try:
                                    yield f'{path}/{member.name}', pq.ParquetFile(stream)
                                except (OSError, ValueError, pa.ArrowException) as exc:
                                    print(f'Skipping unreadable replay-frame Parquet {path}/{member.name}: {exc}',
                                          flush=True)
            except (OSError, tarfile.TarError) as exc:
                print(f'Skipping unreadable replay archive {path}: {exc}', flush=True)

def source_player_slots(available):
    # Return physical source slots, excluding metadata-only ghost slots.

    slots = sorted({
        int(name.split('_')[1]) for name in available
        if name.startswith('player_') and name.endswith('_team_id') and name.split('_')[1].isdigit()
    })

    return tuple(
        slot for slot in slots
        if {
            f'player_{slot}_{feature}'
            for feature in ('team_id', 'on_field', 'demoed', *PLAYER_CORE_FEATURES)
        }.issubset(available)
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
    control_aliases = {
        'throttle': ('throttle', 'control_throttle', 'input_throttle'),
        'steer': ('steer', 'control_steer', 'input_steer'),
        'pitch': ('pitch', 'control_pitch', 'input_pitch', 'pitch_input', 'controls_pitch'),
        'yaw': ('yaw', 'control_yaw', 'input_yaw', 'yaw_input', 'controls_yaw'),
        'roll': ('roll', 'control_roll', 'input_roll', 'roll_input', 'controls_roll'),
        'boost': ('is_boosting', 'control_boost', 'input_boost', 'boost_input', 'boost'),
        'handbrake': ('handbrake', 'control_handbrake', 'input_handbrake'),
    }
    player_columns.update(
        f'player_{slot}_{alias}'
        for slot in slots
        for aliases in control_aliases.values()
        for alias in aliases
        if f'player_{slot}_{alias}' in available
    )
    player_columns.update(
        f'player_{slot}_{feature}'
        for slot in slots
        for feature in ('touching_ball', 'is_touching_ball', 'id')
        if f'player_{slot}_{feature}' in available
    )

    optional = {
        name for name in ('replay_datetime', 'replay_date',
                          'goal_event_id', 'goal_event_team_id',
                          *EVENT_FLAG_COLUMNS)
        if name in available
    }

    boosts = {name for name in available if name.startswith('boost_pickup_')}

    return sorted(required | player_columns | optional | boosts)


def iter_replays(parquet, columns):
    # Yield a contiguous replay while keeping bounded Parquet batches in memory.
    pending = None
    completed = set()

    for batch in parquet.iter_batches(
        batch_size=RECORD_BATCH_ROWS, columns=columns, use_threads=True,
    ):
        frame = pl.from_arrow(batch)
        replay_ids = frame['replay_id'].to_numpy()

        if replay_ids.size == 0:
            continue

        boundaries = np.r_[0, np.flatnonzero(replay_ids[1:] != replay_ids[:-1]) + 1, replay_ids.size]

        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            replay_id = replay_ids[start]
            part = frame.slice(int(start), int(stop - start))

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
    # Avoid rebuilding DataFrame.columns for each of thousands of field reads.
    try:
        series = frame.get_column(column)
    except pl.exceptions.ColumnNotFoundError:
        return np.full(frame.height, default, dtype=np.float32)

    if not series.dtype.is_numeric():
        series = series.cast(pl.Float32, strict=False)

    if series.null_count():
        series = series.fill_null(default)
    # Writable Arrow conversion plus copy=False avoids a second allocation for
    # the common Float32 case while preserving the writable-array contract.
    return series.to_numpy(writable=True).astype(np.float32, copy=False)


def first_event_rows(event_ids):
    """Vectorized equivalent of retaining the first row for each event id."""
    event_ids = np.asarray(event_ids)
    valid_rows = np.flatnonzero(event_ids != '')

    if not valid_rows.size:
        return valid_rows

    _, first = np.unique(event_ids[valid_rows], return_index=True)
    return np.sort(valid_rows[first])


def _init_replay_loader_worker(_worker_id):
    """Avoid nested CPU thread pools inside each PyTorch loader process."""
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    pa.set_cpu_count(1)


def goal_rows_and_teams(frame):
    # Use native goal events, or score increments when historical fields are absent.
    if {'goal_event_id', 'goal_event_team_id'}.issubset(frame.columns):
        event_ids = frame['goal_event_id'].cast(pl.String, strict=False).fill_null('').to_numpy()
        teams = frame['goal_event_team_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
        rows = (event_ids != '') & np.isin(teams, (0, 1))

        if rows.any():
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
    # Choose up to three sustained physical streams per team and strength mask.

    # Metadata-only slots are rejected by ``source_player_slots``. A transient
    # physical stream is ignored unless its on-field exposure is material
    # relative to the least-present selected player. This preserves
    # ordinary 3v3 replays with a ghost/seventh slot while exposing real 4v3,
    # 3v4, and 4v4 states for filtering.

    slots = source_player_slots(set(frame.columns))
    by_team = {0: [], 1: []}
    present_by_slot = {}

    for slot in slots:
        prefix = f'player_{slot}_'
        team_values = frame[prefix + 'team_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
        known = team_values[np.isin(team_values, (0, 1))]

        if not known.size:
            continue

        team = int(np.bincount(known, minlength=2).argmax())
        position = np.column_stack(
            [numeric(frame, prefix + axis, np.nan) for axis in ('pos_x', 'pos_y', 'pos_z')]
        )
        on_field = numeric(frame, prefix + 'on_field', 0.0).astype(bool)
        demoed = numeric(frame, prefix + 'demoed', 0.0).astype(bool)
        active = on_field & ~demoed
        exposure = int((active & np.isfinite(position).all(axis=1)).sum())
        # A demolished car is temporarily off field but still occupies its
        # team's roster slot for match-strength filtering.
        present_by_slot[slot] = on_field | demoed

        if exposure > 0:
            by_team[team].append((slot, exposure))

    if not by_team[0] and not by_team[1]:
        raise ValueError('could not identify a physical player on either team')

    selected = {}
    strength_slots = {0: (), 1: ()}

    for team in (0, 1):
        ranked = sorted(by_team[team], key=lambda item: (-item[1], item[0]))

        if not ranked:
            selected[team] = ()
            continue

        third_exposure = ranked[min(2, len(ranked) - 1)][1]
        selected[team] = tuple(slot for slot, _ in ranked[:3])
        # A fourth genuine participant has substantial exposure; one-frame
        # parser ghosts do not. This set is only used for strength filtering.
        relevance_floor = max(1, int(np.ceil(third_exposure * 0.25)))
        strength_slots[team] = tuple(slot for slot, exposure in ranked if exposure >= relevance_floor)

    blue_strength = sum((present_by_slot[slot].astype(np.int8) for slot in strength_slots[0]),
                        np.zeros(frame.height, dtype=np.int8))
    orange_strength = sum((present_by_slot[slot].astype(np.int8) for slot in strength_slots[1]),
                          np.zeros(frame.height, dtype=np.int8))
    valid_strengths = np.asarray((
        (1, 0), (0, 1),
        (1, 1),
        (1, 2), (2, 1), (2, 2),
        (3, 2), (2, 3), (3, 3),
    ), dtype=np.int8)
    observed_strengths = np.column_stack((blue_strength, orange_strength))
    valid_strength = np.any(
        np.all(observed_strengths[:, None, :] == valid_strengths[None, :, :], axis=2),
        axis=1,
    )
    return selected, valid_strength


def remap_match_roster(frame, selected):
    # Map dynamic source slots to the model’s stable blue-then-orange six slots.
    available = set(frame.columns)
    missing_contact = [source for team in (0, 1) for source in selected[team]
                       if not any(f'player_{source}_{feature}' in available
                                  for feature in ('touching_ball', 'is_touching_ball'))
                       and 'frame_touch_player_slot' not in available]

    if missing_contact:
        raise ValueError(f'missing player touching-ball columns for source slots: {missing_contact}')

    mapping = {
        source: team * MAX_PLAYERS_PER_TEAM + position + 1
        for team in (0, 1)
        for position, source in enumerate(selected[team])
    }
    updates = []

    source_for_destination = {destination: source for source, destination in mapping.items()}

    for destination in PLAYER_SLOTS:
        target_prefix = f'player_{destination}_'
        team = (destination - 1) // MAX_PLAYERS_PER_TEAM
        source = source_for_destination.get(destination)

        if source is None:
            updates.append(pl.lit(2, dtype=pl.Int64).alias(target_prefix + 'team_id'))
            updates.append(pl.lit('', dtype=pl.String).alias(target_prefix + 'id'))
            updates.extend(
                pl.lit(0.0, dtype=pl.Float32).alias(target_prefix + feature)
                for feature in PLAYER_FEATURES
            )
            continue

        source_prefix = f'player_{source}_'
        updates.append(pl.lit(team, dtype=pl.Int64).alias(target_prefix + 'team_id'))
        updates.append((pl.col(source_prefix + 'id').cast(pl.String, strict=False)
                        if source_prefix + 'id' in available else pl.lit('', dtype=pl.String))
                       .alias(target_prefix + 'id'))

        for feature in PLAYER_FEATURES:
            if feature in ('touching_ball', 'is_picking_up_big_boost',
                           'is_picking_up_small_boost'):
                continue

            source_column = source_prefix + feature

            if source_column in available:
                value = pl.col(source_column).cast(pl.Float32, strict=False)
                updates.append(value.alias(target_prefix + feature))
            else:
                updates.append(pl.lit(0.0, dtype=pl.Float32).alias(target_prefix + feature))

        contact_column = next((source_prefix + feature for feature in
                               ('touching_ball', 'is_touching_ball')
                               if source_prefix + feature in available), None)
        contact = (
            pl.col(contact_column).cast(pl.Boolean, strict=False).fill_null(False)
            if contact_column is not None
            else pl.col('frame_touch_player_slot').cast(pl.Int64, strict=False) == source
        )
        updates.append(contact.cast(pl.Float32).alias(target_prefix + 'touching_ball'))

    pickup_columns = tuple(
        name
        for name in available
        if name.startswith('boost_pickup_event_player_') and name.endswith('_slot')
    )
    event_slot_arrays = []

    for column in pickup_columns:
        values = numeric(frame, column, -1.0).astype(np.int64)
        mapped = np.full(values.shape, -1, dtype=np.int64)

        for source, destination in mapping.items():
            mapped[values == source] = destination

        updates.append(pl.Series(column, pa.array(mapped, mask=mapped < 0)))
        event_slot_arrays.append(values)
    # Event rows can fall between 10 Hz training samples. Mark the matching
    # stable player entity so the pickup remains visible at the next sample.
    event_ids = (frame['boost_pickup_event_id'].cast(pl.String, strict=False)
                 .fill_null('').to_numpy() if 'boost_pickup_event_id' in available
                 else np.full(frame.height, '', dtype=object))
    event_kinds = (frame['boost_pickup_type'].cast(pl.String, strict=False)
                   .fill_null('').to_numpy() if 'boost_pickup_type' in available
                   else np.full(frame.height, '', dtype=object))
    first_rows = first_event_rows(event_ids)
    event_slots = (np.column_stack(event_slot_arrays) if event_slot_arrays
                   else np.empty((frame.height, 0), dtype=np.int64))

    for source, destination in mapping.items():
        participant = (np.any(event_slots == source, axis=1) if event_slots.shape[1]
                       else np.zeros(frame.height, dtype=bool))
        event_rows = first_rows[participant[first_rows]]
        big = np.zeros(frame.height, dtype=np.float32)
        small = np.zeros(frame.height, dtype=np.float32)
        big_rows = event_rows[event_kinds[event_rows] == 'big']
        small_rows = event_rows[event_kinds[event_rows] == 'small']
        big[big_rows] = 1.0
        small[small_rows] = 1.0
        updates.extend((
            pl.Series(f'player_{destination}_is_picking_up_big_boost', big),
            pl.Series(f'player_{destination}_is_picking_up_small_boost', small),
        ))

    mapped = frame.with_columns(updates)
    keep_columns = set(OUTPUT_COLUMNS) | {
        'frame_delta', 'frame_seconds_elapsed', 'frame_is_active',
        'blue_team_score', 'orange_team_score',
        'goal_event_id', 'goal_event_team_id', 'boost_pickup_event_id',
        'boost_pickup_type', 'replay_datetime', 'replay_date',
    } | set(pickup_columns)
    return mapped.select(column for column in mapped.columns if column in keep_columns)


def attach_boost_state(frame):
    # Causally reconstruct pad availability without iterating every frame.
    available = set(frame.columns)
    times = frame['_source_time'].to_numpy()
    stints = frame['stint_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    values = np.zeros((frame.height, 34, len(BOOST_FEATURES)), dtype=np.float32)

    values[..., :2] = BOOST_POSITIONS
    # The arena floor is z=0; pad pickup origins sit above it at z=70/73.
    values[:, :, 2] = BOOST_PAD_Z
    values[:, :6, 3] = 1.0
    values[:, 6:, 4] = 1.0
    cooldown = np.r_[np.full(6, 10.0), np.full(28, 4.0)]
    pickup_slots = [
        name
        for name in available
        if name.startswith('boost_pickup_event_player_') and name.endswith('_slot')
    ]
    event_times = [[] for _ in range(34)]
    ready_times = [[] for _ in range(34)]
    stint_starts = np.flatnonzero(np.r_[True, stints[1:] != stints[:-1]])
    # Every pad resets at a stint boundary. Store those reset events once and
    # merge them per pad below instead of running 34 Python appends per stint.
    reset_times = times[stint_starts]

    if {'boost_pickup_event_id', 'boost_pickup_type'}.issubset(available):
        event_ids = (frame['boost_pickup_event_id'].cast(pl.String, strict=False)
                     .fill_null('').to_numpy())
        kinds = frame['boost_pickup_type'].cast(pl.String, strict=False).fill_null('').to_numpy()
        event_rows = first_event_rows(event_ids)
        slots = (np.column_stack([
            numeric(frame, column, -1.0).astype(np.int64) for column in pickup_slots
        ]) if pickup_slots else np.empty((frame.height, 0), dtype=np.int64))
        positions = {
            slot: np.column_stack([numeric(frame, f'player_{slot}_pos_{axis}', np.nan) for axis in 'xyz'])
            for slot in PLAYER_SLOTS
        }

        for index in event_rows:
            kind = kinds[index]

            if kind == 'reset':
                for pad in range(34):
                    event_times[pad].append(times[index])
                    ready_times[pad].append(times[index])

                continue

            if kind not in ('big', 'small'):
                continue

            candidates = np.arange(6) if kind == 'big' else np.arange(6, 34)

            for slot in slots[index]:
                if slot not in PLAYER_SLOTS:
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
        event = np.concatenate((reset_times, np.asarray(event_times[pad], dtype=np.float64)))
        ready = np.concatenate((reset_times, np.asarray(ready_times[pad], dtype=np.float64)))
        order = np.argsort(event, kind='stable')
        event, ready = event[order], ready[order]
        prior = np.searchsorted(event, times, side='right') - 1
        refill = np.zeros(frame.height, dtype=np.float32)
        valid = prior >= 0
        refill[valid] = np.maximum(ready[prior[valid]] - times[valid], 0.0)
        values[:, pad, 6] = refill
        values[:, pad, 5] = refill <= 0.0

    return frame.with_columns(
        pl.Series(name, values.reshape(frame.height, -1)[:, index])
        for index, name in enumerate(BOOST_COLUMNS)
    )


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

        if prefix.startswith('player_'):
            # Replay dodge impulses are world-space vectors; dodge torque is
            # car-relative and stays in the local frame.
            for field in ('dodge_impulse', 'double_jump_impulse'):
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


def randomize_prepared_replay(frame):
    # Choose one field orientation per stint, then rotate cached replay data.
    stints = frame['stint_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    _, group = np.unique(stints, return_inverse=True)
    selected = (np.random.random(group.max() + 1) < 0.5)[group]
    columns = frame.columns
    transformed = apply_random_augmentation(frame.with_columns(
        pl.Series('random_augmentation', selected)
    ))
    return transformed.select(columns)


def prepare_replay(
    frame,
    random_augmentation=False,
    target_sample_rate=TARGET_SAMPLE_RATE,
    inference=False,
    include_replay_date=False,
):
    # Compile one replay with exact targets, dynamic roster mapping, and valid strength states.
    if not np.isfinite(target_sample_rate) or target_sample_rate <= 0 or target_sample_rate > SOURCE_HZ:
        raise ValueError('target_sample_rate must lie in (0, 30]')

    sample_period_frames = SOURCE_HZ / target_sample_rate
    pickup_window_frames = int(np.ceil(sample_period_frames))
    frame_numbers = frame['frame_number'].cast(pl.Int64).to_numpy()

    frame_steps = np.diff(frame_numbers)

    if np.any(frame_steps <= 0):
        frame = frame.sort('frame_number')
        frame_numbers = frame['frame_number'].cast(pl.Int64).to_numpy()

        if np.any(np.diff(frame_numbers) <= 0):
            raise ValueError('frame_number must be strictly increasing within a replay')

    selected, valid_strength = resolve_match_roster(frame)
    frame = remap_match_roster(frame, selected)
    # Preserve pickup events when reducing 30 Hz replay rows to a lower model
    # sample rate: a row marks a pickup in the preceding sampling interval.
    pickup_updates = []

    for feature in ('is_picking_up_big_boost', 'is_picking_up_small_boost'):
        for slot in PLAYER_SLOTS:
            column = f'player_{slot}_{feature}'
            values = frame[column].to_numpy().astype(bool, copy=False)
            pooled = values.copy()

            for offset in range(1, pickup_window_frames):
                pooled[offset:] |= values[:-offset]

            pickup_updates.append(pl.Series(column, pooled.astype(np.float32)))

    if pickup_updates:
        frame = frame.with_columns(pickup_updates)

    available_columns = set(frame.columns)

    source_time = np.cumsum(numeric(frame, 'frame_delta', 0.0), dtype=np.float64)
    goal_row, teams = goal_rows_and_teams(frame)
    stints = frame['stint_id'].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    goal_time = np.full(frame.height, np.nan, dtype=np.float64)
    goal_team = np.full(frame.height, -1, dtype=np.int8)
    _, stint_group = np.unique(stints, return_inverse=True)
    first_goal = np.full(stint_group.max() + 1, frame.height, dtype=np.int64)
    goal_indices = np.flatnonzero(goal_row)
    goal_indices = goal_indices[goal_indices > 0]
    pre_goal_indices = goal_indices - 1
    # A scoring event belongs to the play interval ending at the frame
    # immediately before the goal. This also handles a stint_id change on the
    # goal frame itself.
    np.minimum.at(first_goal, stint_group[pre_goal_indices], goal_indices)
    row_goal = first_goal[stint_group]
    stint_ends_with_goal = row_goal < frame.height
    goal_time[stint_ends_with_goal] = source_time[row_goal[stint_ends_with_goal]]
    goal_team[stint_ends_with_goal] = teams[row_goal[stint_ends_with_goal]]
    has_goal = np.isfinite(goal_time)

    if inference:
        goal_time.fill(np.nan)
        goal_team.fill(-1)
        stint_ends_with_goal.fill(False)
        has_goal.fill(False)

    frame = frame.with_columns([
        pl.Series('_source_time', source_time), pl.Series('stint_id', stints),
        pl.Series('_stint_ends_with_goal', stint_ends_with_goal),
        pl.Series('time_until_next_goal', goal_time - source_time),
        pl.Series('blue_goal_next', has_goal & (goal_team == 0)),
        pl.Series('orange_goal_next', has_goal & (goal_team == 1)),
        pl.Series('_goal_row', goal_row),
    ])
    active = frame['frame_is_active'].cast(pl.Boolean, strict=False).fill_null(False).to_numpy()
    keep = active & valid_strength & ~goal_row
    eligible_rows = np.flatnonzero(keep)
    last_eligible = np.full(first_goal.shape, -1, dtype=np.int64)
    np.maximum.at(last_eligible, stint_group[eligible_rows], eligible_rows)
    anchors = np.where(first_goal < frame.height, first_goal - 1, last_eligible)
    sample = np.zeros(frame.height, dtype=bool)

    if eligible_rows.size:
        frame_offsets = frame_numbers[anchors[stint_group[eligible_rows]]] - frame_numbers[eligible_rows]
        nearest_sample_tick = np.rint(frame_offsets * target_sample_rate / SOURCE_HZ)
        sampled_frame_offsets = np.rint(
            nearest_sample_tick * SOURCE_HZ / target_sample_rate
        ).astype(np.int64)
        sampled_rows = eligible_rows[
            (frame_offsets >= 0) & (frame_offsets == sampled_frame_offsets)
        ]
        sample[sampled_rows] = True

    if frame.is_empty():
        return frame

    current_stints = frame['stint_id'].cast(pl.Int64).fill_null(-1).to_numpy()

    if random_augmentation:
        # All frames in a stint share one orientation so future-state labels
        # remain in the coordinate system of the evaluated frame.
        _, group = np.unique(current_stints, return_inverse=True)
        augmentation = (np.random.random(group.max() + 1) < 0.5)[group]
    else:
        augmentation = np.zeros(frame.height, dtype=bool)

    updates = [
        pl.col('frame_seconds_elapsed').cast(pl.Float32).alias('seconds_elapsed'),
        (300.0 - pl.col('frame_seconds_elapsed').cast(pl.Float32)).clip(lower_bound=0.0).alias('game_seconds_remaining'),
        (pl.col('frame_seconds_elapsed').cast(pl.Float32) >= 300.0).cast(pl.Float32).alias('is_overtime'),
        pl.col('blue_team_score').cast(pl.Float32).alias('blue_score'),
        pl.col('orange_team_score').cast(pl.Float32).alias('orange_score'),
        *(pl.lit(value, dtype=pl.Float32).alias(f'blue_goal_{feature}')
          for feature, value in zip(GOAL_FEATURES, BLUE_GOAL, strict=True)),
        *(pl.lit(value, dtype=pl.Float32).alias(f'orange_goal_{feature}')
          for feature, value in zip(GOAL_FEATURES, ORANGE_GOAL, strict=True)),
        pl.Series(
            'random_augmentation',
            augmentation,
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
    frame = frame.with_columns(
        pl.col(name).cast(pl.Boolean, strict=False).fill_null(False).alias(name)
        if name in available_columns else pl.lit(False).alias(name)
        for name in EVENT_FLAG_COLUMNS
    )
    columns = OUTPUT_COLUMNS

    if include_replay_date:
        columns += tuple(name for name in ('replay_datetime', 'replay_date')
                         if name in frame.columns)
    # Finish feature construction first; active/strength/goal filtering and
    # configured-rate sampling are applied together as the final preprocessing
    # operation. frame_number preserves the source cadence across removed rows.
    return frame.select(columns).filter(pl.Series('_sampled_training_frame', sample))

class LGPSScaler:
    # AnalyzeRL-Boxcars uses raw boost in [0, 255], world units, quaternion
    # components in [-1, 1], and respawn timers in seconds.
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
        tuple(
            5000.0 if name.startswith('pos_') else
            2500.0 if name.startswith('vel_') or 'impulse_' in name else
            360.0 if name.startswith('ang_vel_') else
            255.0 if name == 'boost_raw' else
            3.0 if name == 'respawn_timer' else
            3.0 if name.startswith('dodge_torque_') else 1.0
            for name in PLAYER_FEATURES
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
        (300.0, 1.0, 1.0, 10.0, 10.0),
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

class LGPSEntityTokenizer:
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
        # Convert the batch's numeric inputs in one Polars-to-NumPy pass. The
        # former per-column conversions repeatedly crossed the Python/Arrow
        # boundary and allocated an array for every feature before stacking.
        numeric_columns = tuple(column for column in dict.fromkeys((
            *BALL_FEATURES,
            *(f"player_{slot}_{feature}" for slot in slots if slot is not None
              for feature in PLAYER_FEATURES),
            *(f"player_{slot}_team_id" for slot in slots if slot is not None),
            *CONTEXT_FEATURES,
            *(GOAL_COLUMNS if has_goals else ()),
            *(column for name, _, columns, _ in BOOST_ENTITY_GROUPS
              if name in self.model_entities for column in columns),
        )) if column in frame.columns)
        defaults = {
            f"player_{slot}_team_id": -1.0
            for slot in slots if slot is not None
        }
        numeric_frame = frame.select(
            pl.col(column).cast(pl.Float32, strict=False)
            .fill_null(defaults.get(column, 0.0)).alias(column)
            for column in numeric_columns
        )
        numeric_values = numeric_frame.to_numpy(order="c")
        numeric_index = {name: index for index, name in enumerate(numeric_columns)}

        def columns_as_array(columns):
            present = [column for column in columns if column in numeric_index]

            if len(present) == len(columns):
                return numeric_values[:, [numeric_index[column] for column in columns]]

            result = np.zeros((frame.height, len(columns)), dtype=np.float32)

            for output_index, column in enumerate(columns):
                if column in numeric_index:
                    result[:, output_index] = numeric_values[:, numeric_index[column]]

            return result

        ball = columns_as_array(BALL_FEATURES)

        if all(slot is not None for slot in slots):
            player_columns = tuple(
                f"player_{slot}_{feature}"
                for slot in slots for feature in PLAYER_FEATURES
            )
            players = columns_as_array(player_columns).reshape(
                frame.height, len(slots), len(PLAYER_FEATURES),
            )
        else:
            players = np.zeros(
                (frame.height, len(slots), len(PLAYER_FEATURES)), dtype=np.float32,
            )

            for slot_index, slot in enumerate(slots):
                if slot is not None:
                    players[:, slot_index, :] = columns_as_array(
                        tuple(f"player_{slot}_{feature}" for feature in PLAYER_FEATURES)
                    )

        context = columns_as_array(CONTEXT_FEATURES)

        if not self.include_context:
            context.fill(0.0)

        goals = (
            columns_as_array(GOAL_COLUMNS)
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
            numeric_values[:, numeric_index[f"player_{slot}_team_id"]]
            if slot is not None and f"player_{slot}_team_id" in numeric_index
            else np.full(frame.height, -1, dtype=np.float32)
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
                result[name] = torch.from_numpy(
                    columns_as_array(columns).reshape(
                        frame.height, count, len(BOOST_FEATURES),
                    )
                )

        return result

class LGPSBatchBuilder:
    entity_types = LGPSEntityTokenizer.entity_types
    entity_teams = LGPSEntityTokenizer.entity_teams

    def __init__(self, model_entities="all", *, targets_per_team=1, geometric_width=1.0,
                 horizon_seconds=10.0, horizon_max=30.0,
                 two_way_prediction=False, drop_neither=False,
                 aux_tasks=(), sample_rate=TARGET_SAMPLE_RATE):
        self.model_entities = normalize_model_entities(model_entities)
        self.tokenizer = LGPSEntityTokenizer(model_entities=self.model_entities)
        self.targets_per_team, self.geometric_width = targets_per_team, geometric_width
        self.horizon_seconds = horizon_seconds
        self.horizon_max = horizon_max
        self.two_way_prediction = bool(two_way_prediction)
        self.drop_neither = bool(drop_neither)
        self.aux_tasks = configured_aux_tasks(aux_tasks)
        self.sample_rate = float(sample_rate)

    def filter_indices(self, frame, indices):
        if not self.two_way_prediction and not self.drop_neither:
            return np.asarray(indices, dtype=np.int64)

        indices = (np.arange(indices.start, indices.stop, indices.step, dtype=np.int64)
                   if isinstance(indices, range) else np.asarray(indices, dtype=np.int64))
        times = frame['time_until_next_goal'].to_numpy()[indices].astype(np.float64, copy=False)
        blue = frame['blue_goal_next'].to_numpy()[indices].astype(bool, copy=False)
        orange = frame['orange_goal_next'].to_numpy()[indices].astype(bool, copy=False)
        valid = np.ones(indices.size, dtype=bool)

        if self.drop_neither:
            stint_has_goal = (
                frame['_stint_ends_with_goal']
                .cast(pl.Boolean, strict=False)
                .fill_null(False)
                .to_numpy()[indices]
            )
            valid &= stint_has_goal

        if self.two_way_prediction:
            valid &= np.isfinite(times) & (times > 0.0) & (blue | orange)

        return indices[valid]

    def build(self, frame, indices=None, include_aux=False, *, already_filtered=False):
        indices = range(frame.height) if indices is None else indices

        if not already_filtered:
            indices = self.filter_indices(frame, indices)
        else:
            indices = (np.arange(indices.start, indices.stop, indices.step, dtype=np.int64)
                       if isinstance(indices, range) else np.asarray(indices, dtype=np.int64))

        if not indices.size:
            return None

        result = build_frame_batch(frame, indices,
                             self.tokenizer,
                             targets_per_team=self.targets_per_team,
                             geometric_width=self.geometric_width,
                             horizon_seconds=self.horizon_seconds,
                             horizon_max=self.horizon_max,
                             two_way_prediction=self.two_way_prediction)

        if include_aux and self.aux_tasks:
            result.update(build_aux_targets(
                frame, indices, self.aux_tasks, self.sample_rate,
            ))

        return result

class LGPSReplaySource:
    def __init__(self, path, model_entities="all", sample_rate=TARGET_SAMPLE_RATE,
                 train_split=0.9, val_split=0.0, exclude_paths=()):
        self.model_entities = normalize_model_entities(model_entities)
        path = Path(path)
        excluded_paths = tuple(Path(excluded).resolve() for excluded in exclude_paths)

        def is_excluded(candidate):
            resolved = Path(candidate).resolve()
            return any(resolved.is_relative_to(excluded) for excluded in excluded_paths)

        self._prepared_input = None

        if (not math.isfinite(train_split) or not math.isfinite(val_split)
                or train_split < 0 or val_split < 0 or train_split + val_split > 1):
            raise ValueError('train_split and val_split must be non-negative and sum to at most one')

        if path.is_file():
            candidates = (path,) if path.suffix.lower() in ('.parquet', '.tar') else ()
        elif path.is_dir():
            candidates = tuple(sorted(
                candidate for candidate in path.rglob('*')
                if candidate.is_file() and candidate.suffix.lower() in ('.parquet', '.tar')
                and not is_excluded(candidate)
            ))
        else:
            raise FileNotFoundError(f'LGPS input path does not exist: {path}')

        if not candidates:
            raise FileNotFoundError(f'No replay Parquet files or archives under: {path}')

        # Enumerate once: replay trees can contain tens of thousands of files,
        # and a second recursive walk adds avoidable startup
        # latency on top of the per-file lazy validation in loader workers.
        is_raw = source_is_raw(path, candidates)

        if is_raw:
            self.is_raw = True
            self.raw_path = path
            # Validate each Parquet in its assigned loader worker. A global
            # schema scan over tens of thousands of mounted files adds minutes
            # before the first batch; per-file validation still skips malformed
            # files before decoding rows and preserves the replay cache.
            self.raw_paths = candidates
            self.raw_row_counts = {}

            if not self.raw_paths:
                raise FileNotFoundError(f'No replay Parquet files or archives under: {path}')

            self.sample_rate = sample_rate
            self.train_split = float(train_split)
            self.val_split = float(val_split)
            # Raw replay inputs may be read-only. Keep prepared replay caches
            # in temp storage, keyed by the input path so separate sources do
            # not collide.
            cache_root = Path(os.environ.get('LGPS_CACHE_DIR', Path(tempfile.gettempdir()) / 'lgps_cache'))
            cache_key = hashlib.blake2b(str(path.resolve()).encode('utf-8'), digest_size=12).hexdigest()
            self.cache_dir = cache_root / cache_key
            self.paths = self.raw_paths
            self.read_columns = {}
            self.row_groups = {}
            self.row_counts = {}
            self.total_rows = None
            return

        self.is_raw = False
        candidate_paths = tuple(candidate for candidate in candidates
                                if candidate.suffix.lower() == '.parquet')

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


    def prepared_cache_path(self, source_path, source_name, replay_id):
        # Invalidate cached frames when the source or preparation settings change.
        source_path = Path(source_path)
        stat = source_path.stat()
        identity = '\0'.join((
            str(source_path.resolve()), str(stat.st_size), str(stat.st_mtime_ns),
            str(source_name), str(replay_id), str(self.sample_rate),
            str(PREPARED_CACHE_VERSION),
        ))
        key = hashlib.blake2b(identity.encode('utf-8'), digest_size=20).hexdigest()
        return self.cache_dir / f'{key}.cache'

    def replay_manifest_path(self, source_path):
        source_path = Path(source_path)
        stat = source_path.stat()
        identity = '\0'.join((str(source_path.resolve()), str(stat.st_size),
                              str(stat.st_mtime_ns), str(PREPARED_CACHE_VERSION),
                              str(self.sample_rate)))
        key = hashlib.blake2b(identity.encode('utf-8'), digest_size=20).hexdigest()
        return self.cache_dir / f'{key}.manifest'

    def split_for_replay(self, source_name, replay_id):
        source_parts = str(source_name).replace('\\', '/').lower().split('/')

        if '2026' in source_parts:
            return 'test'

        digest = hashlib.blake2b(str(replay_id).encode('utf-8'), digest_size=8).digest()
        value = int.from_bytes(digest, 'big') / float(1 << 64)

        if value < self.train_split:
            return 'train'

        if value < self.train_split + self.val_split:
            return 'validation'

        return 'test'

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_prepared_input'] = None
        return state

    @staticmethod
    def row_groups_for_path(path):
        # Return row-group starts and lengths without scanning Parquet data.
        reader = pq.ParquetFile(path)

        try:
            metadata = reader.metadata
            start = 0
            groups = []

            for index in range(metadata.num_row_groups):
                length = metadata.row_group(index).num_rows
                groups.append((start, length))
                start += length

            return tuple(groups)
        finally:
            reader.close()

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
            "replay_id", "stint_id", "time_until_next_goal", *GOAL_LABEL_COLUMNS,
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
                f"{path}: missing model-ready columns: {sorted(missing)}; rerun LGPS prepare_dataset"
            )

        optional = tuple(column for column in (
            "_source_time",
            "_stint_ends_with_goal",
            "_next_touch_slot",
            "_next_touch_seconds",
            *(f"player_{slot}_id" for slot in PLAYER_SLOTS),
            *(f"player_{slot}_touching_ball" for slot in PLAYER_SLOTS),
        ) if column in schema)
        return tuple(dict.fromkeys(
            (
            "replay_id", "stint_id", *optional, "time_until_next_goal", *GOAL_LABEL_COLUMNS,
                *goal_columns,
                *(BIG_BOOST_COLUMNS if "big_boosts" in self.model_entities else ()),
                *(SMALL_BOOST_COLUMNS if "small_boosts" in self.model_entities else ()),
                    *CONTEXT_FEATURES,
                *BALL_FEATURES,
                *team_columns,
                *player_columns,
            )
        ))

    def _read_chunks(self, path, start, end, chunk_rows):
        # Read ordinary replay-sized row groups in one call. Stream only large
        # groups so a single group cannot dominate each worker's live memory.
        groups = [index for index, (group_start, group_rows) in enumerate(self.row_groups[path])
                  if group_start < end and group_start + group_rows > start]

        if not groups:
            return

        reader = pq.ParquetFile(path)

        try:
            columns = list(self.read_columns[path])

            for group_index in groups:
                group_start, group_rows = self.row_groups[path][group_index]

                if group_rows <= chunk_rows:
                    table = reader.read_row_group(group_index, columns=columns)
                    first = max(start - group_start, 0)
                    last = min(end - group_start, group_rows)
                    yield pl.from_arrow(table.slice(first, last - first))
                    continue

                cursor = group_start

                for batch in reader.iter_batches(
                    batch_size=chunk_rows, row_groups=[group_index], columns=columns,
                ):
                    batch_start = cursor
                    cursor += batch.num_rows
                    first = max(start - batch_start, 0)
                    last = min(end - batch_start, batch.num_rows)

                    if last > first:
                        yield pl.from_arrow(batch.slice(first, last - first))
        finally:
            reader.close()

    def read_batches(self, path, batch_size=2048, offset=0, length=None, lookahead=0):
        # Yield a contiguous range without materializing whole Parquet row groups.
        end = self.row_counts[path] if length is None else min(
            self.row_counts[path], offset + length
        )

        if offset < 0 or end < offset:
            raise ValueError("invalid parquet row range")

        pending = None

        for current in self._read_chunks(
            path, offset, end, max(batch_size * 16, batch_size + lookahead)
        ):
            pending = current if pending is None else pl.concat([pending, current])

            while pending.height >= batch_size + lookahead:
                yield pending.slice(0, batch_size + lookahead), batch_size
                pending = pending.slice(batch_size)

        while pending is not None and pending.height:
            count = min(batch_size, pending.height)
            yield pending.slice(0, count + lookahead), count
            pending = pending.slice(count)

class LGPSPreparedBatchDataset(IterableDataset):
    def __init__(
        self,
        source,
        builder,
        batch_size,
        rank=0,
        world_size=1,
        split_start=0,
        split_end=None,
        lookahead=0,
        include_aux=False,
    ):
        super().__init__()
        self.source = source
        self.builder = builder
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.split_start = split_start
        self.split_end = source.total_rows if split_end is None else split_end
        self.lookahead = lookahead
        self.include_aux = include_aux

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1
        shard_rank = self.rank * worker_count + worker_id
        shard_count = self.world_size * worker_count
        split_rows = self.split_end - self.split_start
        shard_start = self.split_start + (split_rows * shard_rank) // shard_count
        shard_end = self.split_start + (split_rows * (shard_rank + 1)) // shard_count
        pending_batches = []
        buffered_rows = 0

        def take_batch(target_rows):
            nonlocal buffered_rows
            pieces = {}
            remaining = target_rows

            while remaining:
                batch, offset = pending_batches[0]
                available = batch['temporal_targets'].shape[0] - offset
                take = min(available, remaining)

                for name, value in batch.items():
                    pieces.setdefault(name, []).append(value[offset:offset + take])

                offset += take
                remaining -= take
                buffered_rows -= take

                if offset == batch['temporal_targets'].shape[0]:
                    pending_batches.pop(0)
                else:
                    pending_batches[0] = (batch, offset)

            return {
                name: values[0] if len(values) == 1 else torch.cat(values, dim=0)
                for name, values in pieces.items()
            }

        global_row = 0

        for path in self.source.paths:
            path_rows = self.source.row_counts[path]
            path_start, path_end = global_row, global_row + path_rows
            global_row = path_end

            if path_end <= shard_start or path_start >= shard_end:
                continue

            local_start = max(shard_start - path_start, 0)
            local_end = min(shard_end - path_start, path_rows)
            batches = self.source.read_batches(
                path, batch_size=self.batch_size, offset=local_start,
                length=local_end - local_start, lookahead=self.lookahead,
            )

            for batch in batches:
                frame, count = batch
                prepared = self.builder.build(
                    frame, range(count), include_aux=self.include_aux,
                )

                if prepared is not None:
                    pending_batches.append((prepared, 0))
                    buffered_rows += prepared['temporal_targets'].shape[0]

                    while buffered_rows >= self.batch_size:
                        yield take_batch(self.batch_size)

        if buffered_rows:
            yield take_batch(buffered_rows)


class LGPSRawReplayBatchDataset(IterableDataset):
    def __init__(self, source, builder, batch_size, *, split, rank=0, world_size=1,
                 include_aux=False):
        super().__init__()
        self.source = source
        self.builder = builder
        self.batch_size = batch_size
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.include_aux = include_aux

    def build_batch(self, replays, row_count):
        # Tokenize one full batch instead of tokenizing each replay fragment
        # separately and concatenating dozens of small tensor dictionaries.
        frames = []
        indices = []
        offset = 0
        remaining = row_count
        consumed = []

        for replay in replays:
            frame, eligible, cursor = replay
            take = min(remaining, eligible.size - cursor)

            if take <= 0:
                continue

            frames.append(frame)
            indices.append(eligible[cursor:cursor + take] + offset)
            replay[2] += take
            consumed.append(replay)
            remaining -= take
            offset += frame.height

            if remaining == 0:
                break

        if remaining:
            raise RuntimeError(f'raw LGPS batch is short by {remaining} rows')

        frame = (frames[0] if len(frames) == 1
                 else pl.concat(frames, how='vertical_relaxed', rechunk=False))
        batch = self.builder.build(
            frame, np.concatenate(indices), include_aux=self.include_aux,
            already_filtered=True,
        )

        if any(value.device.type != 'cpu' for value in batch.values()
               if isinstance(value, torch.Tensor)):
            raise RuntimeError('LGPS replay preparation must run on CPU')

        return batch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1
        shard_rank = self.rank * worker_count + worker_id
        shard_count = self.world_size * worker_count
        # Interleave sorted match files across workers without opening every
        # Parquet footer before the first batch is available.
        paths = [path for index, path in enumerate(self.source.raw_paths)
                 if index % shard_count == shard_rank]
        pending_replays = []
        pending_rows = 0

        def emit_prepared(prepared, is_training, eligible=None):
            nonlocal pending_rows

            if is_training and not prepared.is_empty():
                prepared = randomize_prepared_replay(prepared)

            if eligible is None:
                eligible = self.builder.filter_indices(prepared, range(prepared.height))

            if eligible.size:
                pending_replays.append([prepared, eligible, 0])
                pending_rows += eligible.size

            while pending_rows >= self.batch_size:
                batch = self.build_batch(pending_replays, self.batch_size)
                pending_rows -= self.batch_size
                pending_replays[:] = [replay for replay in pending_replays
                                      if replay[2] < replay[1].size]
                yield batch

        if self.split == 'train':
            seed = worker_info.seed if worker_info is not None else torch.initial_seed()
            np.random.default_rng(seed).shuffle(paths)

        for path in paths:
            manifest_path = self.source.replay_manifest_path(path)
            manifest = None

            if manifest_path.exists():
                try:
                    entries = json.loads(manifest_path.read_text(encoding='utf-8'))

                    if all(
                        entry.get('skip', False)
                        or (entry.get('cache') and
                            (self.source.cache_dir / entry['cache']).is_file())
                        for entry in entries
                    ):
                        manifest = entries
                except (OSError, ValueError, KeyError, TypeError):
                    manifest = None

            if manifest is not None:
                for entry in manifest:
                    if entry.get('skip', False):
                        continue

                    replay_split = self.source.split_for_replay(entry['source'], entry['replay_id'])

                    if self.split != 'all' and replay_split != self.split:
                        continue

                    prepared = pl.read_parquet(self.source.cache_dir / entry['cache'])
                    eligible = self.builder.filter_indices(prepared, range(prepared.height))

                    if eligible.size:
                        yield from emit_prepared(
                            prepared,
                            self.split == 'train' and replay_split == 'train',
                            eligible,
                        )

                continue

            entries = []

            for source_name, parquet in iter_parquets(path):
                try:
                    columns = selected_source_columns(
                        set(parquet.schema_arrow.names),
                    )
                except (OSError, ValueError, pa.ArrowException) as exc:
                    if isinstance(exc, ValueError) and not any(
                        reason in str(exc) for reason in (
                            'missing required source columns',
                            'no physical player slots with complete position, velocity, and rotation fields',
                        )
                    ):
                        raise

                    print(
                        f"Skipping incompatible replay-frame Parquet {source_name}: {exc}",
                        flush=True,
                    )
                    continue

                for replay in iter_replays(parquet, columns):
                    replay_id = str(replay['replay_id'][0])
                    cache_path = self.source.prepared_cache_path(path, source_name, replay_id)

                    if cache_path.exists():
                        prepared = pl.read_parquet(cache_path)
                    else:
                        prepared = prepare_replay(
                            replay,
                            random_augmentation=False,
                            target_sample_rate=self.source.sample_rate,
                        )

                    eligible = self.builder.filter_indices(prepared, range(prepared.height))

                    if not eligible.size:
                        entries.append({'source': source_name, 'replay_id': replay_id,
                                        'skip': True})
                        continue

                    if not cache_path.exists():
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary_path = cache_path.with_name(
                            f'{cache_path.name}.{os.getpid()}.{worker_id}.tmp'
                        )
                        # This cache is scratch data for training. Snappy trades
                        # slightly larger local files for much cheaper CPU encode/decode.
                        pq.write_table(prepared.to_arrow(), temporary_path,
                                       compression='snappy', use_dictionary=False)

                        try:
                            os.replace(temporary_path, cache_path)
                        except OSError:
                            if not cache_path.exists():
                                raise

                            temporary_path.unlink(missing_ok=True)

                    entries.append({'source': source_name, 'replay_id': replay_id,
                                    'cache': cache_path.name})
                    replay_split = self.source.split_for_replay(source_name, replay_id)

                    if self.split == 'all' or replay_split == self.split:
                        yield from emit_prepared(
                            prepared, self.split == 'train' and replay_split == 'train',
                            eligible,
                        )

            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_manifest = manifest_path.with_name(
                f'{manifest_path.name}.{os.getpid()}.{worker_id}.tmp'
            )
            temporary_manifest.write_text(json.dumps(entries), encoding='utf-8')
            os.replace(temporary_manifest, manifest_path)

        if pending_rows:
            yield self.build_batch(pending_replays, pending_rows)



class LGPSDataset:
    # Data pipeline assembled from training, model, and runtime contracts.
    def __init__(self, train_config, model_config, runtime_config, rank=0, world_size=1):
        self.train_config = train_config
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.rank = rank
        self.world_size = world_size
        self.aux_task_settings = configured_aux_tasks(train_config.aux_tasks)
        excluded_paths = ()

        if train_config.test_path is not None:
            train_root = Path(train_config.input_path).resolve()
            test_root = Path(train_config.test_path).resolve()

            if train_root != test_root and test_root.is_relative_to(train_root):
                excluded_paths = (test_root,)

        self.source = LGPSReplaySource(train_config.input_path, model_config.model_entities,
                                     model_config.sample_rate,
                                     train_config.train_split, train_config.val_split,
                                     exclude_paths=excluded_paths)
        self.test_source = (
            LGPSReplaySource(train_config.test_path, model_config.model_entities,
                           model_config.sample_rate)
            if train_config.test_path is not None else None
        )
        self.builder = LGPSBatchBuilder(
            model_config.model_entities,
            targets_per_team=model_config.targets_per_team,
            geometric_width=model_config.geometric_width,
            horizon_seconds=model_config.horizon_seconds,
            horizon_max=model_config.horizon_max,
            two_way_prediction=model_config.two_way_prediction,
            drop_neither=train_config.drop_neither,
            aux_tasks=train_config.aux_tasks,
            sample_rate=model_config.sample_rate,
        )

        if self.source.is_raw:
            self.ranges = None
        else:
            distributed = self.world_size > 1 and dist.is_available() and dist.is_initialized()

            if not distributed or self.rank == 0:
                replays = (pl.concat([pl.scan_parquet(path).select('replay_id')
                                      for path in self.source.paths])
                           .with_row_index('_row')
                           .filter(pl.col('replay_id').ne(pl.col('replay_id').shift()).fill_null(True))
                           .collect(engine='streaming'))

                if replays['replay_id'].null_count() or replays['replay_id'].n_unique() != replays.height:
                    raise ValueError('replay_id must be non-null and contiguous')

                boundaries = (*replays['_row'].to_list(), self.source.total_rows)
                replay_count = len(boundaries) - 1
                train_count = min(replay_count, int(math.floor(replay_count * train_config.train_split + 0.5)))
                validation_count = min(replay_count, int(math.floor(
                    replay_count * (train_config.train_split + train_config.val_split) + 0.5
                )))
                cuts = (boundaries[train_count], boundaries[validation_count])
            else:
                cuts = (0, 0)

            if distributed:
                device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else torch.device('cpu')
                shared_cuts = torch.tensor(cuts, dtype=torch.int64, device=device)
                dist.broadcast(shared_cuts, src=0)
                cuts = tuple(int(value) for value in shared_cuts.tolist())

            train_end, validation_end = cuts
            self.ranges = {
                "train": (0, train_end),
                "validation": (train_end, validation_end),
                "test": ((0, self.test_source.total_rows) if self.test_source is not None
                         else (validation_end, self.source.total_rows)),
            }

    def loader(self, batch_size, split):
        source = self.test_source if split == 'test' and self.test_source is not None else self.source

        if source.is_raw:
            raw_split = 'all' if source is not self.source else split
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
                    worker_init_fn=_init_replay_loader_worker,
                )

                if 'in_order' in inspect.signature(DataLoader).parameters:
                    kwargs['in_order'] = False

            dataset = LGPSRawReplayBatchDataset(
                source, self.builder, batch_size, split=raw_split,
                rank=self.rank, world_size=self.world_size,
                include_aux=(split == 'train' and bool(self.train_config.aux_tasks)),
            )
            return DataLoader(dataset, **kwargs)

        if self.source.is_raw:
            split_start, split_end = 0, source.total_rows
        else:
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
                worker_init_fn=_init_replay_loader_worker,
            )

            if 'in_order' in inspect.signature(DataLoader).parameters:
                kwargs['in_order'] = False

        dataset = LGPSPreparedBatchDataset(
            source,
            self.builder,
            batch_size,
            rank=self.rank,
            world_size=self.world_size,
            split_start=split_start,
            split_end=split_end,
            lookahead=self.auxiliary_lookahead() if split == 'train' else 0,
            include_aux=(split == 'train' and bool(self.train_config.aux_tasks)),
        )
        return DataLoader(
            dataset,
            **kwargs,
        )

    def auxiliary_lookahead(self):
        settings = self.aux_task_settings.get(LGPSBallStateAux)
        return max(settings.future_frames, default=0) if settings is not None else 0

    def train_loader(self):
        return self.loader(self.runtime_config.per_gpu_batch_size or self.train_config.batch_size, "train")

    def validation_loader(self):
        batch_size = self.runtime_config.per_gpu_batch_size or self.train_config.batch_size
        return self.loader(min(self.train_config.validation_batch_size, batch_size), "validation")

    def test_loader(self):
        batch_size = self.runtime_config.per_gpu_batch_size or self.train_config.batch_size
        return self.loader(min(self.train_config.validation_batch_size, batch_size), "test")

def _source_parquet_specs(source):
    source = Path(source)
    paths = ((source,) if source.is_file() else tuple(sorted(
        path for path in source.rglob('*')
        if path.is_file() and path.suffix.lower() in ('.parquet', '.tar')
    )))
    specs = []

    for path in paths:
        if path.suffix.lower() == '.parquet':
            specs.append((str(path), None))
        elif path.suffix.lower() == '.tar':
            with tarfile.open(path, 'r:') as archive:
                specs.extend((str(path), member.name) for member in archive
                             if member.isfile() and member.name.endswith('_frames.parquet'))

    return specs


def _source_parquet(source_path, member_name=None):
    if member_name is None:
        return pq.ParquetFile(source_path), None

    archive = tarfile.open(source_path, 'r:')
    stream = archive.extractfile(member_name)

    if stream is None:
        archive.close()
        raise FileNotFoundError(f'{member_name} is missing from {source_path}')

    return pq.ParquetFile(stream), archive


def _date_year(value):
    import datetime
    import re

    if value is None:
        return None

    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.year

    match = re.match(r'^(20\d{2})', str(value))
    return int(match.group(1)) if match else None


def _prepare_source_shard(task):
    (index, source_specs, staging_dir, rate, test_replay_ids, test_year) = task
    source_writers = {}
    source_buffers = {'train': [], 'test': []}
    source_buffered_rows = {'train': 0, 'test': 0}
    shard_paths = {}
    replay_info = []

    def flush_source(split):
        if not source_buffers[split]:
            return

        table = pa.concat_tables(source_buffers[split])

        if split not in source_writers:
            shard_path = str(Path(staging_dir) / f'{index:08d}-{split}.parquet')
            shard_paths[split] = shard_path
            float_columns = [field.name for field in table.schema
                             if pa.types.is_floating(field.type)]
            source_writers[split] = pq.ParquetWriter(
                shard_path, table.schema,
                compression='zstd', use_dictionary=['replay_id'],
                use_byte_stream_split=float_columns,
            )

        source_writers[split].write_table(table, row_group_size=WRITE_BATCH_ROWS)
        source_buffers[split].clear()
        source_buffered_rows[split] = 0

    try:
        for source_path, member_name in source_specs:
            parquet, archive = _source_parquet(source_path, member_name)

            try:
                columns = selected_source_columns(set(parquet.schema_arrow.names))

                for replay in iter_replays(parquet, columns):
                    replay_id = replay['replay_id'][0]
                    date_col = next((name for name in ('replay_datetime', 'replay_date')
                                     if name in replay.columns), None)
                    year = _date_year(replay[date_col][0]) if date_col is not None else None
                    replay_info.append((replay_id, year))

                    if test_year is not None:
                        if year is None:
                            raise ValueError(f'replay {replay_id} has no parseable date for test_year split')

                        split = 'test' if year == test_year else 'train'
                    else:
                        split = 'test' if test_replay_ids is not None and replay_id in test_replay_ids else 'train'

                    prepared = prepare_replay(
                        replay, random_augmentation=True,
                        target_sample_rate=rate,
                    )

                    if prepared.is_empty():
                        continue

                    table = prepared.to_arrow()
                    source_buffers[split].append(table)
                    source_buffered_rows[split] += table.num_rows

                    if source_buffered_rows[split] >= WRITE_BATCH_ROWS:
                        flush_source(split)
            finally:
                if archive is not None:
                    archive.close()

        for split in source_buffers:
            flush_source(split)
    finally:
        for writer in source_writers.values():
            writer.close()

    return index, shard_paths, replay_info


def prepare_dataset(source, output_dir, *, max_replays=None, target_sample_rate=TARGET_SAMPLE_RATE,
                    test_replay_ids=None,
                    workers=1, staging_dir=None, test_year=None):
    # Accept a Parquet file, tar archive, or nested source directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    seen_replays = set()
    test_replay_ids = None if test_replay_ids is None else set(test_replay_ids)
    workers = max(1, int(workers))

    if test_year is not None and test_replay_ids is not None:
        raise ValueError('specify either test_year or test_replay_ids, not both')

    if workers > 1 and max_replays is None:
        return _prepare_dataset_parallel(
            source, output_dir, target_sample_rate=target_sample_rate,
            test_replay_ids=test_replay_ids, test_year=test_year,
            workers=workers, staging_dir=staging_dir,
        )

    try:
        for _, parquet in iter_parquets(Path(source)):
            columns = selected_source_columns(set(parquet.schema_arrow.names))

            for replay in iter_replays(parquet, columns):
                replay_id = replay['replay_id'][0]

                if replay_id in seen_replays:
                    raise ValueError(f'duplicate replay_id across source files: {replay_id}')

                seen_replays.add(replay_id)

                if test_year is not None:
                    date_col = next((name for name in ('replay_datetime', 'replay_date')
                                     if name in replay.columns), None)
                    replay_year = _date_year(replay[date_col][0]) if date_col is not None else None

                    if replay_year is None:
                        raise ValueError(f'replay {replay_id} has no parseable replay date for test_year split')

                    split = 'test' if replay_year == test_year else 'train'
                else:
                    split = 'test' if test_replay_ids is not None and replay_id in test_replay_ids else 'train'

                prepared = prepare_replay(replay, random_augmentation=True,
                                          target_sample_rate=target_sample_rate)

                if not prepared.is_empty():
                    table = prepared.rechunk().to_arrow()

                    if split not in writers:
                        writers[split] = pq.ParquetWriter(
                            output_dir / f'{split}.parquet', table.schema,
                            compression='zstd', use_dictionary=False,
                        )

                    writers[split].write_table(table, row_group_size=WRITE_BATCH_ROWS)

                if max_replays is not None and len(seen_replays) >= max_replays:
                    break

            if max_replays is not None and len(seen_replays) >= max_replays:
                break
    finally:
        for writer in writers.values():
            writer.close()

    return {split: output_dir / f'{split}.parquet' for split in writers}


def _prepare_dataset_parallel(source, output_dir, *, target_sample_rate,
                              test_replay_ids, test_year, workers, staging_dir):
    from joblib import Parallel, delayed

    specs = _source_parquet_specs(source)

    if not specs:
        raise FileNotFoundError(f'No replay Parquet files or archives under: {source}')

    stage_root = Path(staging_dir) if staging_dir is not None else output_dir
    stage_root.mkdir(parents=True, exist_ok=True)
    counts = {'train': 0, 'test': 0}
    seen_replays = set()
    writers = {}
    table_buffers = {'train': [], 'test': []}
    buffered_rows = {'train': 0, 'test': 0}

    def flush_tables(split):
        if not table_buffers[split]:
            return

        table = pa.concat_tables(table_buffers[split])

        if split not in writers:
            float_columns = [field.name for field in table.schema
                             if pa.types.is_floating(field.type)]
            writers[split] = pq.ParquetWriter(
                output_dir / f'{split}.parquet', table.schema,
                compression='zstd', use_dictionary=['replay_id'],
                use_byte_stream_split=float_columns,
            )

        writers[split].write_table(table, row_group_size=WRITE_BATCH_ROWS)
        table_buffers[split].clear()
        buffered_rows[split] = 0

    with tempfile.TemporaryDirectory(prefix='lgps-parallel-', dir=stage_root) as temporary:
        sources_per_task = max(1, min(512, workers * 128))
        source_batches = (
            specs[offset:offset + sources_per_task]
            for offset in range(0, len(specs), sources_per_task)
        )
        tasks = (
            (index, source_batch, temporary, target_sample_rate,
             test_replay_ids, test_year)
            for index, source_batch in enumerate(source_batches)
        )
        pending = {}
        next_index = 0
        build_started = time.perf_counter()
        reported_replays = 0
        results = Parallel(
            n_jobs=workers, backend='loky', batch_size=1,
            return_as='generator_unordered', pre_dispatch=workers,
            backend_kwargs={'inner_max_num_threads': 1},
        )(delayed(_prepare_source_shard)(task) for task in tasks)

        try:
            for index, shard_paths, replay_info in results:
                pending[index] = (shard_paths, replay_info)

                while next_index in pending:
                    shard_paths, replay_info = pending.pop(next_index)

                    for replay_id, year in replay_info:
                        if replay_id in seen_replays:
                            raise ValueError(f'duplicate replay_id across source files: {replay_id}')

                        seen_replays.add(replay_id)

                        if test_year is not None and year is None:
                            raise ValueError(f'replay {replay_id} has no parseable date for test_year split')

                        split = ('test' if year == test_year else 'train') if test_year is not None else (
                            'test' if test_replay_ids is not None and replay_id in test_replay_ids else 'train'
                        )
                        counts[split] += 1

                    for split, shard in shard_paths.items():
                        try:
                            table = pq.read_table(shard)
                            table_buffers[split].append(table)
                            buffered_rows[split] += table.num_rows

                            if buffered_rows[split] >= WRITE_BATCH_ROWS:
                                flush_tables(split)
                        finally:
                            Path(shard).unlink(missing_ok=True)

                    next_index += 1
                    replay_count = len(seen_replays)

                    if replay_count - reported_replays >= 500:
                        elapsed = time.perf_counter() - build_started
                        print(f'LGPSDataset: {replay_count:,} replays, '
                              f'{replay_count / elapsed:.2f} replays/s')
                        reported_replays = replay_count

            for split in table_buffers:
                flush_tables(split)
        finally:
            for writer in writers.values():
                writer.close()

    if test_year is not None and (counts['train'] == 0 or counts['test'] == 0
                                  or set(writers) != {'train', 'test'}):
        raise RuntimeError(f'parallel dataset split is empty or has no rows: {counts}')

    elapsed = time.perf_counter() - build_started
    print(f'LGPSDataset: {len(seen_replays):,} replays, '
          f'{len(seen_replays) / elapsed:.2f} replays/s total')
    return {split: output_dir / f'{split}.parquet' for split in writers}


def source_is_raw(path, candidates=None):
    # Prepared inputs have model targets; raw replay frames have source frame fields.
    path = Path(path)

    if candidates is not None:
        paths = tuple(candidates)
    elif path.is_file():
        paths = (path,)
    elif path.is_dir():
        paths = tuple(sorted(
            candidate for candidate in path.rglob('*')
            if candidate.is_file() and candidate.suffix.lower() in ('.parquet', '.tar')
        ))
    else:
        paths = ()

    if not paths:
        return False

    candidate = paths[0]

    if candidate.suffix.lower() == '.tar':
        return True

    schema = set(pq.ParquetFile(candidate).schema_arrow.names)
    return {'frame_number', 'frame_is_active', 'stint_id'}.issubset(schema)


def cleanup_training_cache(train_config):
    # Delete temporary replay caches after a training run.
    paths = (train_config.input_path, train_config.test_path)
    cache_dirs = set()

    for value in paths:
        if value is None:
            continue

        path = Path(value)

        if source_is_raw(path):
            cache_root = Path(os.environ.get('LGPS_CACHE_DIR', Path(tempfile.gettempdir()) / 'lgps_cache'))
            cache_key = hashlib.blake2b(str(path.resolve()).encode('utf-8'), digest_size=12).hexdigest()
            cache_dirs.add(cache_root / cache_key)

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
