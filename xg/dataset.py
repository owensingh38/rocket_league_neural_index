from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from .config import FRAMES_FOLDER, MODEL_SEGMENTS

FRAMES_FOLDER = str(FRAMES_FOLDER)
SEGMENT_TEAM_SIZES = {'duels': 1, 'doubles': 2, 'standard': 3}
RANDOM_STATE = 42
MAX_PLAYERS = 8
GOAL_LINK_MAX_FRAMES = 120
BACK_NET_Y = 6000.0
GOAL_CENTER_TO_POST = 892.755
GOAL_HEIGHT = 642.775
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
CEILING_Z = 2044.0

GRAVITY = 650.0
BALL_RADIUS = 92.75
CAR_MIN_INTERCEPT_SPEED = 500.0
CAR_MAX_SPEED = 2300.0
CAR_GROUND_SPEED_CAP = 1410.0
CAR_GROUND_THROTTLE_ACCEL = 1600.0
CAR_AIR_THROTTLE_ACCEL = 66.667
BOOST_ACCEL_GROUND = 991.667
BOOST_ACCEL_AIR = 1058.333
BOOST_CONSUMPTION_PER_SECOND = 33.333
CAR_MAX_YAW_RATE = 5.5

TRAJECTORY_SECONDS = 5.0
TRAJECTORY_SAMPLES = 12
AERIAL_INTERCEPT_Z = 300.0
GOAL_PLANE_Y = BACK_WALL_Y
HISTORY_HALF_LIFE_SECONDS = 8.0
DOUBLE_TAP_SECONDS = 5.0
DOUBLE_TAP_BACK_WALL_DISTANCE = 900.0
DOUBLE_TAP_BACK_WALL_PROJECTION_SECONDS = 3.5
WALL_SHOT_DISTANCE = 350.0
CEILING_SHOT_DISTANCE = 350.0

EVENT_TYPES = {
    1: 'kickoff', 2: 'game_join', 3: 'game_leave', 4: 'shot', 5: 'save',
    6: 'goal', 7: 'assist', 8: 'pass', 9: 'missed_pass', 10: 'missed_shot',
    11: 'air_dribble', 12: 'ground_dribble', 13: 'flick', 14: 'demo',
    15: 'bump', 16: 'entry', 17: 'exit', 18: 'retrieval', 19: 'boost_pickup',
    20: 'flip_reset', 21: 'press', 22: 'shadow', 23: 'whiff', 24: 'fake',
    25: 'challenge', 26: 'multi_commit', 27: 'respawn', 28: 'turnover',
}

EVENT_TYPE_IDS = {event_type: f'{event_id:02}' for event_id, event_type in EVENT_TYPES.items()}

LAST_FRAME_CONTEXT_EVENTS = {
    'shot', 'save', 'pass', 'missed_pass', 'missed_shot',
    'air_dribble', 'ground_dribble', 'flick', 'demo', 'bump',
    'retrieval', 'boost_pickup', 'flip_reset', 'press', 'shadow',
    'fake', 'challenge', 'multi_commit', 'turnover',
}

EVENT_PLAYER_WIDTHS = {
    'game_join': MAX_PLAYERS,
    'game_leave': MAX_PLAYERS,
    'kickoff': 2,
    'goal': 2,
    'assist': 2,
    'pass': 2,
    'missed_pass': 2,
    'demo': 2,
    'bump': MAX_PLAYERS,
    'turnover': 2,
    'boost_pickup': MAX_PLAYERS,
    'press': 2,
    'shadow': 2,
    'fake': 2,
    'challenge': MAX_PLAYERS,
    'multi_commit': MAX_PLAYERS,
}

HISTORY_EVENT_TYPES = (
    'turnover', 'pass', 'shot', 'save',
    'demo', 'bump', 'challenge', 'entry', 'exit', 'retrieval',
    'whiff', 'shadow', 'press', 'fake', 'missed_pass', 'missed_shot',
    'multi_commit', 'respawn',
)

OFF_COLUMNS = (
    'off_demo', 'off_kickoff', 'off_challenge_win', 'off_bump',
    'off_controlled_entry', 'off_controlled_exit', 'off_retrieval',
    'off_uncontrolled_entry', 'off_uncontrolled_exit', 'off_air_dribble',
    'off_ground_dribble', 'off_flick', 'off_pass', 'off_fake', 'off_whiff',
    'off_flip_reset', 'off_wall', 'off_ceiling',
)

PLAYER_STATE_FIELDS = (
    'on_field', 'demoed', 'touching_ball',
    'throttle', 'steer', 'handbrake',
    'is_driving', 'is_powersliding', 'is_throttling', 'in_reverse',
    'throttle_scaled', 'steer_scaled', 'boost_scaled',
    'supersonic',
    'is_boosting',
    'dodge_torque_x', 'dodge_torque_y', 'dodge_torque_z',
    'dodge_impulse_x', 'dodge_impulse_y', 'dodge_impulse_z',
    'double_jump_impulse_x', 'double_jump_impulse_y', 'double_jump_impulse_z',
    'in_air', 'on_ground', 'on_ceiling', 'on_backwall', 'on_sidewall',
    'pitch', 'yaw', 'roll',
    'is_jumping', 'is_single_jumping', 'is_double_jumping', 'is_dodging',
    'has_dodge', 'jumped', 'double_jumped', 'dodged',
    'pos_x', 'pos_y', 'pos_z',
    'vel_x', 'vel_y', 'vel_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    'rot_x', 'rot_y', 'rot_z',
    'rot_w', 'boost_raw',
)

SHOOTER_FIELDS = (*PLAYER_STATE_FIELDS,)
PASSER_FIELDS = (*PLAYER_STATE_FIELDS,)
DEFENDER_FIELDS = (*PLAYER_STATE_FIELDS,)

PLAYER_FIELDS = (
    'id', 'team',
    *dict.fromkeys((*SHOOTER_FIELDS, *DEFENDER_FIELDS)),
)

PLAYER_BASE_MODEL_FIELDS = {
    'on_field', 'pos_x', 'pos_y', 'pos_z',
    'vel_x', 'vel_y', 'vel_z', 'boost_raw',
}

PLAYER_QUATERNION_FIELDS = ('rot_x', 'rot_y', 'rot_z', 'rot_w')

PLAYER_WORLD_XY_FIELDS = {
    'ang_vel_x', 'ang_vel_y',
    'dodge_impulse_x', 'dodge_impulse_y',
    'double_jump_impulse_x', 'double_jump_impulse_y',
}

PLAYER_FEATURE_ALIASES = {
    'ang_vel_x': 'angular_velocity_x',
    'ang_vel_y': 'angular_velocity_y',
    'ang_vel_z': 'angular_velocity_z',
    'rot_x': 'rotation_x',
    'rot_y': 'rotation_y',
    'rot_z': 'rotation_z',
    'rot_w': 'rotation_w',
}

BASE_SOURCE_COLUMNS = (
    'replay_id', 'frame_number', 'frame_seconds_elapsed', 'event_type', 'event_id',
    'frame_strength_state', 'replay_playlist_label',
    'shot_event_id', 'goal_event_id', 'pass_event_id',
    'frame_touch_team_id', 'frame_touch_player_slot',
    'event_player_1_slot',
    *(
        f'{event_type}_event_player_{event_player}_slot'
        for event_type in ('shot', 'goal', 'pass')
        for event_player in range(1, MAX_PLAYERS + 1)
    ),
    'ball_pos_x', 'ball_pos_y', 'ball_pos_z',
    'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
    'ball_ang_vel_x', 'ball_ang_vel_y', 'ball_ang_vel_z',
)

def _normalize_segments(model_segments: Sequence[str] | str | None) -> tuple[str, ...]:
    if model_segments is None:
        return MODEL_SEGMENTS
    values = (model_segments,) if isinstance(model_segments, str) else tuple(model_segments)
    segments = tuple(dict.fromkeys(str(value).lower() for value in values))
    invalid = sorted(set(segments) - set(MODEL_SEGMENTS))
    if invalid:
        raise ValueError(f'Unknown model segments: {invalid}. Expected values from {MODEL_SEGMENTS}')
    if not segments:
        raise ValueError('model_segments must contain at least one segment')
    return segments


def _frame_paths(frames_folder: str | Path) -> list[Path]:
    paths = sorted(Path(frames_folder).rglob('*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No parquet files found in frames folder: {frames_folder}')
    return paths


def _source_columns(include_event_context: bool = True) -> list[str]:
    columns = [
        *BASE_SOURCE_COLUMNS,
        *(
            f'player_{player_number}_{field}'
            for player_number in range(1, MAX_PLAYERS + 1)
            for field in PLAYER_FIELDS
        ),
    ]
    if include_event_context:
        columns.extend(('stint_id', 'possession_id', 'possession_team'))
    return columns


@lru_cache(maxsize=65_536)
def _read_parquet_schema(path: str) -> pa.Schema:
    # Reuse metadata reads when a source batch feeds multiple lazy branches.
    return pq.read_schema(path)


def _projected_schema(paths: Sequence[Path], columns: set[str]) -> pa.Schema:
    # Raw frame exports use a stable schema within each source season. Unify
    # projected fields from a spread of representative files instead of opening
    # every Parquet footer in a 1k-file batch; absent fields are inserted as nulls.
    schemas = []
    sample_count = min(32, len(paths))
    sample_indices = np.linspace(0, len(paths) - 1, sample_count, dtype=np.int64)
    for index in np.unique(sample_indices):
        path = paths[int(index)]
        schema = _read_parquet_schema(str(path))
        fields = []
        for column in schema.names:
            if column not in columns:
                continue
            field = schema.field(column)
            # Arrow 26 may infer ``string_view`` from one Parquet file and
            # ``large_string`` from another; unify_schemas cannot promote the
            # view type. Normalize views before constructing the dataset schema.
            is_string_view = getattr(pa.types, 'is_string_view', lambda _: False)
            if is_string_view(field.type):
                field = pa.field(field.name, pa.string(), nullable=field.nullable, metadata=field.metadata)
            fields.append(field)
        schemas.append(pa.schema(fields))
    return pa.unify_schemas(schemas, promote_options='permissive')


def _scan_filtered_batches(
    paths: Sequence[Path],
    requested: Sequence[str],
    row_filter: pl.Expr,
    selected_replay_ids: Sequence[str] | None,
) -> pl.LazyFrame:
    schema = _projected_schema(paths, set(requested))
    available = set(schema.names)
    # Use Polars' native Parquet scanner so predicate and projection pushdown
    # happen before Arrow materializes wide player/physics columns.
    scan_schema = {
        field.name: pl.from_arrow(pa.array([], type=field.type)).dtype
        for field in schema
    }
    frame = pl.scan_parquet(
        [str(path) for path in paths],
        schema=scan_schema,
        missing_columns='insert',
        extra_columns='ignore',
        parallel='auto',
        use_statistics=True,
    )
    frame = frame.select(
        pl.col(column) if column in available else pl.lit(None).alias(column)
        for column in requested
    )
    if selected_replay_ids is not None and 'replay_id' in available:
        frame = frame.filter(
            pl.col('replay_id').cast(pl.String, strict=False).is_in(selected_replay_ids)
        )
    return frame.filter(row_filter)


def _scan_shot_goal_batches(
    paths: Sequence[Path],
    selected_replay_ids: Sequence[str] | None,
    include_event_context: bool,
) -> pl.LazyFrame:
    schema_names = set(pq.read_schema(paths[0]).names)
    if 'event_type' in schema_names:
        row_filter = (
            pl.col('event_type').cast(pl.String, strict=False).is_in(['shot', 'goal'])
            | pl.col('frame_touch_player_slot').is_not_null()
        )
    elif {'shot_event_id', 'goal_event_id'} & schema_names:
        event_filters: list[pl.Expr] = []
        if 'shot_event_id' in schema_names:
            event_filters.append(pl.col('shot_event_id').is_not_null())
        if 'goal_event_id' in schema_names:
            event_filters.append(pl.col('goal_event_id').is_not_null())
        event_filters.append(pl.col('frame_touch_player_slot').is_not_null())
        row_filter = pl.any_horizontal(event_filters)
    elif 'event_id' in schema_names:
        row_filter = pl.col('event_id').is_not_null()
    else:
        raise ValueError('No event frame rows matched the requested replays')
    return _scan_filtered_batches(
        paths,
        _source_columns(include_event_context),
        row_filter,
        selected_replay_ids,
    )


def _scan_replay_mode_batches(
    paths: Sequence[Path],
    selected_replay_ids: Sequence[str] | None,
) -> pl.LazyFrame:
    return _scan_filtered_batches(
        paths,
        ['replay_id', 'frame_strength_state', 'replay_playlist_label'],
        pl.col('replay_id').is_not_null(),
        selected_replay_ids,
    )


def _pass_source_columns() -> list[str]:
    return [
        'replay_id', 'frame_number', 'frame_seconds_elapsed', 'event_type', 'event_id',
        'pass_event_id', 'stint_id',
        'event_player_1_slot',
        *(f'pass_event_player_{event_player}_slot' for event_player in range(1, MAX_PLAYERS + 1)),
        'ball_pos_x', 'ball_pos_y', 'ball_pos_z',
        'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
        *(
            f'player_{player_number}_{field}'
            for player_number in range(1, MAX_PLAYERS + 1)
            for field in PASSER_FIELDS
        ),
        *(
            f'player_{player_number}_team'
            for player_number in range(1, MAX_PLAYERS + 1)
        ),
    ]


def _scan_pass_batches(
    paths: Sequence[Path],
    selected_replay_ids: Sequence[str] | None,
) -> pl.LazyFrame:
    schema_names = set(pq.read_schema(paths[0]).names)
    if 'event_type' in schema_names:
        row_filter = pl.col('event_type').cast(pl.String, strict=False) == 'pass'
    elif 'pass_event_id' in schema_names:
        row_filter = pl.col('pass_event_id').is_not_null()
    elif 'event_id' in schema_names:
        row_filter = pl.col('event_id').is_not_null()
    else:
        raise ValueError('No event frame rows matched the requested replays')
    return _scan_filtered_batches(
        paths,
        _pass_source_columns(),
        row_filter,
        selected_replay_ids,
    )


def _timeline_columns() -> list[str]:
    return list(dict.fromkeys([
        'replay_id', 'frame_number', 'frame_seconds_elapsed', 'event_type', 'event_id',
        'event_start_frame', 'event_end_frame', 'event_duration',
        'stint_id', 'possession_id',
        'ball_pos_x', 'ball_pos_y', 'ball_pos_z',
        'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
        'possession_team', 'contested_possession', 'event_player_1_slot',
        *(
            column
            for event_type in EVENT_TYPES.values()
            for column in (
                f'{event_type}_event_id',
                f'{event_type}_event_player_1_slot',
            )
        ),
        *(
            f'player_{player_number}_rotation_role'
            for player_number in range(1, MAX_PLAYERS + 1)
        ),
    ]))


def _scan_timeline_batches(
    paths: Sequence[Path],
    selected_replay_ids: Sequence[str] | None,
) -> pl.DataFrame:
    schema_names = set(pq.read_schema(paths[0]).names)
    if 'event_type' in schema_names:
        row_filter = pl.col('event_type').is_not_null()
    else:
        event_columns = [
            f'{event_type}_event_id'
            for event_type in EVENT_TYPES.values()
            if f'{event_type}_event_id' in schema_names
        ]
        if event_columns:
            row_filter = pl.any_horizontal(pl.col(column).is_not_null() for column in event_columns)
        elif 'event_id' in schema_names:
            row_filter = pl.col('event_id').is_not_null()
        else:
            raise ValueError('No event frame rows matched the requested replays')
        if 'possession_id' in schema_names:
            row_filter = row_filter | (pl.col('possession_id').cast(pl.Int64, strict=False) != 0)
    return _scan_filtered_batches(
        paths,
        _timeline_columns(),
        row_filter,
        selected_replay_ids,
    ).collect(engine='streaming')


def _event_id_expr(event_type: str) -> pl.Expr:
    type_id = EVENT_TYPE_IDS[event_type]
    return pl.coalesce([
        pl.when(pl.col('event_type') == event_type).then(pl.col('event_id')),
        pl.col(f'{event_type}_event_id'),
        pl.col('event_id')
        .cast(pl.String, strict=False)
        .str.extract(rf'(?:^|\|)([019]{type_id}\d+)(?:\||$)', 1),
    ]).cast(pl.String, strict=False)


def _event_player_slot_expr(event_type: str) -> pl.Expr:
    return pl.coalesce([
        pl.when(pl.col('event_type') == event_type).then(pl.col('event_player_1_slot')),
        pl.col(f'{event_type}_event_player_1_slot'),
        pl.when(_event_id_expr(event_type).is_not_null()).then(pl.col('event_player_1_slot')),
    ]).cast(pl.Int8, strict=False)


def _event_player_width(event_type: str) -> int:
    return EVENT_PLAYER_WIDTHS.get(event_type, 1)


def _player_team_for_slot_expr(slot_expr: pl.Expr) -> pl.Expr:
    expression = pl.lit(None, dtype=pl.String)
    for player_number in range(1, MAX_PLAYERS + 1):
        expression = (
            pl.when(slot_expr == player_number)
            .then(pl.col(f'player_{player_number}_team'))
            .otherwise(expression)
        )
    return expression


def _event_player_slot_for_ordinal_expr(
    event_type: str,
    event_column: str,
    ordinal_column: str,
) -> pl.Expr:
    width = _event_player_width(event_type)
    max_instances = max(1, MAX_PLAYERS // width)
    expression = pl.col('event_player_1_slot').cast(pl.Int8, strict=False)
    event_team = _event_team_expr(event_column)
    for ordinal in reversed(range(max_instances)):
        source_slot = 1 if event_type in {'pass', 'missed_pass'} else ordinal * width + 1
        source = pl.col(f'{event_type}_event_player_{source_slot}_slot').cast(pl.Int8, strict=False)
        fallback = pl.col(f'{event_type}_event_player_1_slot').cast(pl.Int8, strict=False)
        candidates = (
            [
                pl.when(_player_team_for_slot_expr(source) == event_team).then(source),
                pl.when(_player_team_for_slot_expr(fallback) == event_team).then(fallback),
                pl.col('event_player_1_slot'),
            ]
            if event_type == 'shot'
            else [source, fallback, pl.col('event_player_1_slot')]
        )
        expression = (
            pl.when(pl.col(ordinal_column) == ordinal)
            .then(pl.coalesce(candidates))
            .otherwise(expression)
        )
    return expression.cast(pl.Int8, strict=False)


def _explode_event_ids(frame: pl.LazyFrame, event_type: str, event_column: str) -> pl.LazyFrame:
    source_row = f'_{event_type}_source_row'
    ordinal = f'_{event_type}_ordinal'
    return (
        frame
        .with_columns([
            pl.int_range(0, pl.len()).alias(source_row),
            pl.col(event_column).cast(pl.String, strict=False).str.split('|').alias(event_column),
        ])
        .explode(event_column)
        .with_columns(
            (pl.col(event_column).cum_count().over(source_row) - 1)
            .cast(pl.Int64)
            .alias(ordinal)
        )
        .with_columns(
            _event_player_slot_for_ordinal_expr(event_type, event_column, ordinal)
            .alias(f'_{event_type}_player_slot')
        )
    )


def _event_team_expr(event_column: str) -> pl.Expr:
    team_id = pl.col(event_column).str.slice(0, 1)
    return (
        pl.when(team_id == '0').then(pl.lit('blue'))
        .when(team_id == '1').then(pl.lit('orange'))
        .otherwise(None)
    )


def _event_type_name_expr() -> pl.Expr:
    expression = pl.col('event_type').cast(pl.String, strict=False)
    for event_id, event_type in reversed(tuple(EVENT_TYPES.items())):
        expression = (
            pl.when(pl.col('_event_type_id') == event_id)
            .then(pl.lit(event_type))
            .otherwise(expression)
        )
    return expression


def _timeline_event_player_slot_expr() -> pl.Expr:
    expression = pl.col('event_player_1_slot').cast(pl.Int8, strict=False)
    for event_id, event_type in reversed(tuple(EVENT_TYPES.items())):
        expression = (
            pl.when(pl.col('_event_type_id') == event_id)
            .then(pl.coalesce([
                pl.col(f'{event_type}_event_player_1_slot'),
                pl.col('event_player_1_slot'),
            ]))
            .otherwise(expression)
        )
    return expression.cast(pl.Int8, strict=False)


def _slot_expr(
    field: str,
    dtype: pl.DataType,
    slot_column: str = 'event_player_1_slot',
) -> pl.Expr:
    expression = pl.lit(None, dtype=dtype)
    for player_number in range(1, MAX_PLAYERS + 1):
        expression = (
            pl.when(pl.col(slot_column) == player_number)
            .then(pl.col(f'player_{player_number}_{field}').cast(dtype, strict=False))
            .otherwise(expression)
        )
    return expression


def _segment_expr(team_size_column: str = 'model_team_size') -> pl.Expr:
    team_size = pl.col(team_size_column).cast(pl.Int8, strict=False)
    expression = pl.lit(None, dtype=pl.String)
    for segment, size in reversed(tuple(SEGMENT_TEAM_SIZES.items())):
        expression = pl.when(team_size == size).then(pl.lit(segment)).otherwise(expression)
    return expression


def _playlist_segment_expr() -> pl.Expr:
    label = pl.col('replay_playlist_label').cast(pl.String, strict=False).str.to_lowercase()
    expression = pl.lit(None, dtype=pl.String)
    for segment in MODEL_SEGMENTS:
        expression = pl.when(label == segment).then(pl.lit(segment)).otherwise(expression)
    return expression


def _strength_team_size_expr() -> pl.Expr:
    strength = pl.col('frame_strength_state').cast(pl.String, strict=False)
    left = strength.str.extract(r'^(\d+)v\d+$', 1).cast(pl.Int8, strict=False)
    right = strength.str.extract(r'^\d+v(\d+)$', 1).cast(pl.Int8, strict=False)
    return pl.max_horizontal(left, right)


def _timeline_events(timeline: pl.DataFrame) -> pl.DataFrame:
    # PBP rows are already event-wise; frame rows are normalized from event-owned columns.
    if timeline['event_type'].null_count() < timeline.height:
        events = (
            timeline
            .lazy()
            .filter(pl.col('event_type').is_not_null() & pl.col('event_id').is_not_null())
            .with_columns([
                pl.col('event_id').cast(pl.String, strict=False).alias('_event_key'),
                pl.col('event_player_1_slot').cast(pl.Int8, strict=False),
            ])
        )
    elif 'event_id' in timeline.columns and timeline['event_id'].null_count() < timeline.height:
        events = (
            timeline
            .lazy()
            .filter(pl.col('event_id').is_not_null())
            .with_columns([
                pl.col('event_id')
                .cast(pl.String, strict=False)
                .str.split('|')
                .alias('_event_key'),
                pl.col('event_player_1_slot')
                .cast(pl.Int8, strict=False)
                .alias('event_player_1_slot'),
                pl.lit(None, dtype=pl.String).alias('event_type'),
            ])
            .explode('_event_key')
        )
    else:
        event_frames = []
        for event_type in EVENT_TYPES.values():
            event_id_column = f'{event_type}_event_id'
            event_player_column = f'{event_type}_event_player_1_slot'
            event_frames.append(
                timeline
                .lazy()
                .filter(pl.col(event_id_column).is_not_null())
                .with_columns([
                    pl.col(event_id_column)
                    .cast(pl.String, strict=False)
                    .str.split('|')
                    .alias('_event_key'),
                    pl.col(event_player_column)
                    .cast(pl.Int8, strict=False)
                    .alias('event_player_1_slot'),
                    pl.lit(event_type).alias('event_type'),
                ])
                .explode('_event_key')
            )
        events = pl.concat(event_frames, how='diagonal_relaxed')
    events = (
        events
        .filter(pl.col('_event_key').str.len_chars() >= 3)
        .with_columns([
            pl.col('_event_key').str.slice(1, 2).cast(pl.Int8, strict=False).alias('_event_type_id'),
            pl.col('_event_key').str.slice(0, 1).alias('_event_team_id'),
        ])
        .with_columns(_event_type_name_expr().alias('event_type'))
    )
    slot = _timeline_event_player_slot_expr()
    role = pl.lit(None, dtype=pl.Float32)
    for player_number in range(1, MAX_PLAYERS + 1):
        role = (
            pl.when(slot == player_number)
            .then(pl.col(f'player_{player_number}_rotation_role').cast(pl.Float32, strict=False))
            .otherwise(role)
        )
    normalized = (
        events
        .with_columns([
            slot.alias('_event_player_slot'),
            role.alias('_event_rotation_role'),
            pl.col('stint_id').cast(pl.Int64, strict=False).alias('_event_stint_id'),
            pl.when(pl.col('_event_team_id') == '0').then(pl.lit('blue'))
            .when(pl.col('_event_team_id') == '1').then(pl.lit('orange'))
            .otherwise(None).alias('_event_team'),
            pl.coalesce([
                pl.col('frame_seconds_elapsed').cast(pl.Float32, strict=False),
                pl.col('frame_number').cast(pl.Float32, strict=False) / 30.0,
            ]).alias('_event_seconds'),
        ])
        # Keep first-frame context while retaining the full ID-owned event span.
        .with_columns([
            pl.coalesce([
                pl.col('event_start_frame').cast(pl.Int64, strict=False),
                pl.col('frame_number').min().over(['replay_id', '_event_key']),
            ]).alias('_event_start_frame'),
            pl.coalesce([
                pl.col('event_end_frame').cast(pl.Int64, strict=False),
                pl.col('frame_number').max().over(['replay_id', '_event_key']),
            ]).alias('_event_end_frame'),
            pl.col('_event_seconds').max().over(['replay_id', '_event_key']).alias('_event_end_seconds'),
        ])
        .sort(['replay_id', '_event_key', 'frame_number'])
    )
    first_frame_events = (
        normalized
        .filter(~pl.col('event_type').is_in(LAST_FRAME_CONTEXT_EVENTS))
        .unique(subset=['replay_id', '_event_key'], keep='first', maintain_order=True)
    )
    last_frame_events = (
        normalized
        .filter(pl.col('event_type').is_in(LAST_FRAME_CONTEXT_EVENTS))
        .unique(subset=['replay_id', '_event_key'], keep='last', maintain_order=True)
    )
    return (
        pl.concat([first_frame_events, last_frame_events], how='diagonal_relaxed')
        .sort(['replay_id', 'frame_number', '_event_key'])
        .unique(subset=['replay_id', '_event_key'], keep='first', maintain_order=True)
        .collect(engine='streaming')
    )


def _near_offensive_back_wall(event: Mapping[str, Any], team: str) -> bool:
    ball_y = event.get('ball_pos_y')
    if ball_y is None or not np.isfinite(ball_y):
        return False
    offensive_y = -BACK_WALL_Y if team == 'orange' else BACK_WALL_Y
    if abs(ball_y - offensive_y) <= DOUBLE_TAP_BACK_WALL_DISTANCE:
        return True
    ball_vy = event.get('ball_vel_y')
    if ball_vy is None or not np.isfinite(ball_vy) or abs(ball_vy) <= 1e-6:
        return False
    time_to_wall = (offensive_y - ball_y) / ball_vy
    if not 0.0 < time_to_wall <= DOUBLE_TAP_BACK_WALL_PROJECTION_SECONDS:
        return False
    ball_x = event.get('ball_pos_x')
    ball_z = event.get('ball_pos_z')
    ball_vx = event.get('ball_vel_x')
    ball_vz = event.get('ball_vel_z')
    if any(value is None or not np.isfinite(value) for value in (ball_x, ball_z, ball_vx, ball_vz)):
        return False
    projected_x = ball_x + ball_vx * time_to_wall
    projected_z = ball_z + ball_vz * time_to_wall - 0.5 * GRAVITY * time_to_wall ** 2
    return abs(projected_x) <= SIDE_WALL_X and 0.0 <= projected_z <= CEILING_Z


def _event_context_features(timeline: pl.DataFrame, shots: pl.DataFrame) -> pl.DataFrame:
    events = _timeline_events(timeline)
    events_by_replay: dict[str, list[dict[str, Any]]] = {}
    for event in events.iter_rows(named=True):
        event_type = event.get('event_type') or EVENT_TYPES.get(event['_event_type_id'])
        if event_type is None:
            continue
        event['event_type'] = event_type
        events_by_replay.setdefault(event['replay_id'], []).append(event)

    # Possession IDs own their full frame span, independent of intervening event rows.
    possession_starts = {
        (row['replay_id'], row['possession_id']): row['_possession_seconds']
        for row in (
            events
            .lazy()
            .filter(pl.col('possession_id').cast(pl.Int64, strict=False).fill_null(0) != 0)
            .with_columns(pl.col('_event_seconds').alias('_possession_seconds'))
            .group_by(['replay_id', 'possession_id'])
            .agg(pl.col('_possession_seconds').min())
            .collect(engine='streaming')
            .iter_rows(named=True)
        )
    }

    records = []
    for shot in shots.iter_rows(named=True):
        replay_events = events_by_replay.get(shot['replay_id'], [])
        shot_frame = shot['_frame_order']
        shot_seconds = shot.get('frame_seconds_elapsed')
        if shot_seconds is None or not np.isfinite(shot_seconds):
            shot_seconds = shot_frame / 30.0
        shooter_slot = shot.get('event_player_1_slot')
        shooter_team = shot.get('_shot_team')
        possession_id = shot.get('possession_id')
        possession_team = str(shot.get('possession_team') or '').lower()
        shooter_team_has_possession = (
            possession_id not in (None, 0)
            and possession_team == shooter_team
        )
        possession_start = possession_starts.get((shot['replay_id'], possession_id))
        shooter_team_possession_length = (
            max(shot_seconds - possession_start, 0.0)
            if shooter_team_has_possession and possession_start is not None
            else 0.0
        )
        shot_stint_id = shot.get('stint_id')
        stint_events = [
            event
            for event in replay_events
            if event.get('_event_stint_id') == shot_stint_id
        ]
        # Stint IDs are the parser-owned kickoff-to-kickoff history boundary.
        history = [event for event in stint_events if event['frame_number'] < shot_frame]
        weighted = {event_type: 0.0 for event_type in HISTORY_EVENT_TYPES}
        kickoff_seconds = next(
            (
                event.get('_event_seconds')
                for event in stint_events
                if event['event_type'] == 'kickoff'
                and event.get('_event_seconds') is not None
            ),
            None,
        )
        for event in history:
            event_seconds = event.get('_event_seconds')
            if event_seconds is None or not np.isfinite(event_seconds):
                continue
            if event['event_type'] == 'boost_pickup':
                continue
            event_end_frame = event.get('_event_end_frame')
            if event_end_frame is not None and shot_frame <= event_end_frame:
                weight = 1.0
            else:
                decay_start_frame = (
                    event_end_frame
                    if event_end_frame is not None and np.isfinite(event_end_frame)
                    else event.get('frame_number')
                )
                weight = 0.5 ** (
                    max((shot_frame - decay_start_frame) / 30.0, 0.0)
                    / HISTORY_HALF_LIFE_SECONDS
                )
            if event['event_type'] in weighted:
                weighted[event['event_type']] += weight

        # Offensive context uses short event windows immediately preceding the shot.
        flags = {column: False for column in OFF_COLUMNS}
        recent = []
        for event in reversed(history):
            event_seconds = event.get('_event_seconds')
            if event_seconds is None or not np.isfinite(event_seconds):
                continue
            delta = shot_seconds - event_seconds
            if delta > DOUBLE_TAP_SECONDS:
                break
            if delta > 0.0:
                recent.append((event, delta))
        for event, delta in recent:
            event_type = event['event_type']
            same_team = event.get('_event_team') == shooter_team
            same_shooter = event.get('_event_player_slot') == shooter_slot
            flags['off_demo'] |= delta <= 2.0 and event_type == 'demo' and same_team
            flags['off_kickoff'] |= delta <= 5.0 and event_type == 'kickoff'
            flags['off_challenge_win'] |= delta <= 5.0 and event_type == 'challenge' and same_team
            flags['off_bump'] |= delta <= 2.0 and event_type == 'bump' and same_team
            flags['off_pass'] |= delta <= 5.0 and event_type == 'pass' and same_team
            flags['off_fake'] |= delta <= 5.0 and event_type == 'fake' and same_team
            flags['off_whiff'] |= delta <= 5.0 and event_type == 'whiff' and not same_team
            flags['off_air_dribble'] |= delta <= 3.0 and event_type == 'air_dribble' and same_shooter
            flags['off_ground_dribble'] |= delta <= 3.0 and event_type == 'ground_dribble' and same_shooter
            flags['off_flick'] |= delta <= 1.0 and event_type == 'flick' and same_shooter
            flags['off_flip_reset'] |= delta <= 2.0 and event_type == 'flip_reset' and same_shooter
            if delta <= 2.0 and same_team and event_type in {'entry', 'exit', 'retrieval'}:
                controlled = (
                    str(event.get('possession_team', '')).lower() == shooter_team
                    and not bool(event.get('contested_possession'))
                )
                if event_type == 'retrieval':
                    flags['off_retrieval'] = True
                elif event_type == 'entry':
                    flags['off_controlled_entry' if controlled else 'off_uncontrolled_entry'] = True
                else:
                    flags['off_controlled_exit' if controlled else 'off_uncontrolled_exit'] = True

        ball_x, ball_y, ball_z = shot.get('ball_pos_x'), shot.get('ball_pos_y'), shot.get('ball_pos_z')
        shooter_x = shooter_y = shooter_z = None
        if shooter_slot is not None:
            shooter_x = shot.get(f'player_{int(shooter_slot)}_pos_x')
            shooter_y = shot.get(f'player_{int(shooter_slot)}_pos_y')
            shooter_z = shot.get(f'player_{int(shooter_slot)}_pos_z')
        flags['off_wall'] = any(
            value is not None and np.isfinite(value) and distance <= WALL_SHOT_DISTANCE
            for value, distance in (
                (ball_x, abs(SIDE_WALL_X - abs(ball_x))) if ball_x is not None else (None, np.inf),
                (ball_y, abs(BACK_WALL_Y - abs(ball_y))) if ball_y is not None else (None, np.inf),
                (shooter_x, abs(SIDE_WALL_X - abs(shooter_x))) if shooter_x is not None else (None, np.inf),
                (shooter_y, abs(BACK_WALL_Y - abs(shooter_y))) if shooter_y is not None else (None, np.inf),
            )
        )
        flags['off_ceiling'] = any(
            value is not None and np.isfinite(value) and CEILING_Z - value <= CEILING_SHOT_DISTANCE
            for value in (ball_z, shooter_z)
        )
        flags['off_pass'] |= shot.get('_pass_event_id') is not None
        record = {
            'replay_id': shot['replay_id'],
            '_shot_event_id': shot['_shot_event_id'],
            'history_seconds_since_kickoff': (
                shot_seconds - kickoff_seconds if kickoff_seconds is not None else None
            ),
            'shooter_team_has_possession': shooter_team_has_possession,
            'shooter_team_possession_length': shooter_team_possession_length,
            **flags,
        }
        record.update({
            f'history_weighted_{event_type}_count': weighted[event_type]
            for event_type in HISTORY_EVENT_TYPES
        })
        records.append(record)
    if records:
        return pl.DataFrame(records)
    return pl.DataFrame(
        schema={
            'replay_id': pl.String,
            '_shot_event_id': pl.String,
            'history_seconds_since_kickoff': pl.Float32,
            'shooter_team_has_possession': pl.Boolean,
            'shooter_team_possession_length': pl.Float32,
            **{column: pl.Boolean for column in OFF_COLUMNS},
            **{
                f'history_weighted_{event_type}_count': pl.Float32
                for event_type in HISTORY_EVENT_TYPES
            },
        }
    )


def _pass_context_rows(
    paths: Sequence[Path],
    selected_replay_ids: Sequence[str] | None,
) -> pl.LazyFrame:
    # Pass rows retain their own event identity before nearest-shot linkage.
    return (
        _scan_pass_batches(paths, selected_replay_ids)
        .with_columns([
            _event_id_expr('pass').alias('_pass_event_id'),
            pl.col('frame_number').cast(pl.Int64, strict=False).alias('_pass_frame_order'),
            pl.col('stint_id').cast(pl.Int64, strict=False).alias('_pass_stint_id'),
        ])
        .pipe(_explode_event_ids, 'pass', '_pass_event_id')
        .with_columns(_event_team_expr('_pass_event_id').alias('_pass_team'))
        .with_columns([
            *(
                _slot_expr(field, pl.Float32, '_pass_player_slot')
                .alias(f'_passer_{field}')
                for field in PASSER_FIELDS
            ),
        ])
        .filter(pl.col('_pass_event_id').is_not_null())
        .sort(['replay_id', '_pass_event_id', '_pass_frame_order'])
        .unique(subset=['replay_id', '_pass_event_id'], keep='last', maintain_order=True)
        .select([
            'replay_id', '_pass_event_id', '_pass_team', '_pass_stint_id',
            '_pass_frame_order',
            pl.coalesce([
                pl.col('frame_seconds_elapsed').cast(pl.Float32, strict=False),
                pl.col('_pass_frame_order').cast(pl.Float32, strict=False) / 30.0,
            ]).alias('_pass_seconds'),
            *(
                pl.col(f'ball_{state}_{axis}').cast(pl.Float32, strict=False)
                .alias(f'_pass_ball_{state}_{axis}')
                for state in ('pos', 'vel')
                for axis in ('x', 'y', 'z')
            ),
            *(pl.col(f'_passer_{field}') for field in PASSER_FIELDS),
        ])
    )


def _goal_labels_from_nearest_shot(
    shots: pl.LazyFrame,
    goal_frames: pl.LazyFrame,
) -> pl.LazyFrame:
    # Match goals to the nearest preceding first shot row by the recorded
    # scorer on the scoring team. No separate ball-touch validation is used.
    return (
        goal_frames
        .filter(pl.col('_goal_event_id').is_not_null())
        .with_columns([
            _event_team_expr('_goal_event_id').alias('_goal_team'),
            _event_player_slot_expr('goal').alias('_goal_scorer_slot'),
        ])
        .select(['replay_id', '_goal_event_id', '_goal_team', '_goal_scorer_slot', '_frame_order'])
        .join(
            shots.select([
                'replay_id', '_shot_team', '_shot_event_id', '_shot_player_slot',
                pl.col('_frame_order').alias('_shot_frame'),
            ]),
            left_on=['replay_id', '_goal_team', '_goal_scorer_slot'],
            right_on=['replay_id', '_shot_team', '_shot_player_slot'],
            how='inner',
        )
        .with_columns((pl.col('_frame_order') - pl.col('_shot_frame')).alias('_goal_shot_delta'))
        .filter(
            (pl.col('_goal_shot_delta') >= 0)
            & (pl.col('_goal_shot_delta') <= GOAL_LINK_MAX_FRAMES)
        )
        .sort(['replay_id', '_goal_event_id', '_goal_shot_delta', '_shot_event_id'])
        .unique(subset=['replay_id', '_goal_event_id'], keep='first', maintain_order=True)
        .select(['replay_id', '_shot_event_id'])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias('is_goal'))
    )


def _scan_frame_paths(
    paths: Sequence[Path],
    model_segments: Sequence[str] | str | None,
    include_event_context: bool = True,
    selected_replay_ids: Sequence[str] | None = None,
) -> pl.LazyFrame:
    # Scan a bounded path group into reduced shot-event rows.
    segments = _normalize_segments(model_segments)
    base_frames = (
        _scan_shot_goal_batches(paths, selected_replay_ids, include_event_context)
        .with_columns([
            _event_id_expr('shot').alias('_shot_event_id'),
            _event_id_expr('goal').alias('_goal_event_id'),
            _event_id_expr('pass').alias('_pass_event_id'),
            pl.col('frame_number').cast(pl.Int64, strict=False).alias('_frame_order'),
        ])
        .cache()
    )
    shot_frames = _explode_event_ids(base_frames, 'shot', '_shot_event_id')
    goal_frames = _explode_event_ids(base_frames, 'goal', '_goal_event_id')
    # Modal frame strength owns private replay mode; labels are an explicit fallback.
    replay_segments = (
        _scan_replay_mode_batches(paths, selected_replay_ids)
        .with_columns(_strength_team_size_expr().alias('strength_team_size'))
        .group_by(['replay_id', 'frame_strength_state', 'strength_team_size'])
        .agg([
            pl.len().alias('strength_frame_count'),
            pl.col('replay_playlist_label').drop_nulls().first().alias('replay_playlist_label'),
        ])
        .sort(
            ['replay_id', 'strength_frame_count', 'frame_strength_state'],
            descending=[False, True, False],
        )
        .unique(subset=['replay_id'], keep='first', maintain_order=True)
        .rename({'frame_strength_state': 'replay_strength_state'})
        .with_columns([
            _playlist_segment_expr().alias('playlist_segment'),
            _segment_expr('strength_team_size').alias('strength_segment'),
        ])
        .with_columns(
            pl.coalesce([
                pl.col('strength_segment'),
                pl.col('playlist_segment'),
            ]).alias('model_segment')
        )
        .filter(pl.col('model_segment').is_in(segments))
    )
    # The first row of each shot event is the shooter's last touch before the
    # shot is recorded. Use that row for both features and goal matching.
    shots = (
        shot_frames
        .filter(pl.col('_shot_event_id').is_not_null())
        .with_columns([
            _event_team_expr('_shot_event_id').alias('_shot_team'),
            pl.col('_shot_player_slot').alias('event_player_1_slot'),
            _slot_expr('id', pl.String, '_shot_player_slot').alias('shooter_id'),
        ])
        .sort(['replay_id', '_shot_event_id', '_frame_order'])
        .unique(subset=['replay_id', '_shot_event_id'], keep='first', maintain_order=True)
        .join(replay_segments, on='replay_id', how='inner')
    )
    goals = _goal_labels_from_nearest_shot(shots, goal_frames)
    prepared_shots = (
        shots
        .join(goals, on=['replay_id', '_shot_event_id'], how='left')
        .with_columns([
            pl.lit('shot').alias('event_type'),
            pl.col('is_goal').fill_null(0),
            pl.col('_shot_event_id').alias('event_id'),
            pl.col('_shot_event_id').alias('shot_event_id'),
            pl.col('event_player_1_slot').alias('shot_event_player_1_slot'),
            pl.col('_shot_team').alias('shooter_team'),
        ])
        .drop([
            column
            for column in (
                '_shot_source_row',
                '_shot_ordinal',
                'strength_frame_count',
                'goal_event_id',
                'pass_event_id',
                *(
                    f'{event_type}_event_player_{event_player}_slot'
                    for event_type in ('shot', 'goal', 'pass')
                    for event_player in range(1, MAX_PLAYERS + 1)
                ),
            )
            if column in shots.collect_schema().names()
        ])
    )
    if not include_event_context:
        return prepared_shots
    try:
        pass_context = _pass_context_rows(paths, selected_replay_ids)
        prepared_shots = (
            prepared_shots
            .drop('_pass_event_id')
            .sort(['replay_id', '_frame_order'])
            .join_asof(
                pass_context.sort(['replay_id', '_pass_frame_order']),
                left_on='_frame_order',
                right_on='_pass_frame_order',
                by_left=['replay_id', '_shot_team', 'stint_id'],
                by_right=['replay_id', '_pass_team', '_pass_stint_id'],
                strategy='backward',
                tolerance=GOAL_LINK_MAX_FRAMES,
                check_sortedness=False,
            )
        )
    except ValueError as error:
        if 'No event frame rows matched' not in str(error):
            raise
        # No-pass batches still own the typed pass schema required by segment concatenation.
        prepared_shots = prepared_shots.with_columns([
            pl.lit(None, dtype=pl.Float32).alias('_pass_seconds'),
            *(
                pl.lit(None, dtype=pl.Float32).alias(f'_pass_ball_{state}_{axis}')
                for state in ('pos', 'vel')
                for axis in ('x', 'y', 'z')
            ),
            *(
                pl.lit(None, dtype=pl.Float32).alias(f'_passer_{field}')
                for field in PASSER_FIELDS
            ),
        ])
    shot_rows = prepared_shots.collect(engine='streaming')
    if shot_rows.is_empty():
        return shot_rows.lazy()
    timeline = _scan_timeline_batches(paths, selected_replay_ids)
    context = _event_context_features(timeline, shot_rows)
    return shot_rows.join(
        context,
        on=['replay_id', '_shot_event_id'],
        how='left',
    ).lazy()


def scan_frame_parquets(
    frames_folder: str | Path = FRAMES_FOLDER,
    model_segments: Sequence[str] | str | None = MODEL_SEGMENTS,
    limit: int | None = None,
    random_state: int = RANDOM_STATE,
    include_event_context: bool = True,
) -> pl.LazyFrame:
    # Lazily expose reduced shot rows from a bounded replay sample.
    paths = _frame_paths(frames_folder)
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError('limit must be a positive integer when provided')
        if limit < len(paths):
            rng = np.random.default_rng(random_state)
            sample_size = min(len(paths), limit + max(10, limit // 10))
            indices = np.sort(rng.choice(len(paths), size=sample_size, replace=False))
            paths = [paths[index] for index in indices]
    return _scan_frame_paths(paths, model_segments, include_event_context)


def load_training_frames(
    frames_folder: str | Path = FRAMES_FOLDER,
    model_segments: Sequence[str] | str | None = MODEL_SEGMENTS,
    limit: int | None = None,
    random_state: int = RANDOM_STATE,
    include_event_context: bool = True,
) -> pl.DataFrame:
    # Collect the lazily reduced shot-event training rows as Polars.
    frames = scan_frame_parquets(
        frames_folder,
        model_segments,
        limit,
        random_state,
        include_event_context,
    ).collect(engine='streaming')
    return limit_replays(frames, limit, random_state)


def load_training_parquet(
    training_parquet: str | Path,
    segment: str | None = None,
    limit: int | None = None,
    random_state: int = RANDOM_STATE,
) -> pl.DataFrame:
    # Load the compiled, model-ready xG parquet without reopening raw frames.
    path = Path(training_parquet)
    if not path.is_file():
        raise FileNotFoundError(f'Compiled xG parquet not found: {path}')
    frames = pl.read_parquet(path)
    required = {'replay_id', 'event_id', 'is_goal', 'model_segment'}
    missing = required.difference(frames.columns)
    if missing:
        raise ValueError(f'{path}: missing compiled xG columns: {sorted(missing)}')
    if segment is not None:
        segment = _normalize_segments((segment,))[0]
        frames = frames.filter(pl.col('model_segment').cast(pl.String, strict=False) == segment)
    if frames.is_empty():
        return frames
    if frames.select(pl.struct('replay_id', 'event_id').n_unique()).item() != frames.height:
        raise ValueError(f'{path}: duplicate replay/event identities')
    return limit_replays(frames, limit, random_state)


def load_segment_training_frames(
    frames_folder: str | Path,
    segment: str,
    limit: int | None,
    random_state: int,
    include_event_context: bool = True,
    frame_paths: Sequence[Path] | None = None,
    selected_replay_ids: Sequence[str] | None = None,
    scan_batch_size: int | None = None,
) -> pl.DataFrame:
    # Soft-scan replay files until one segment reaches its replay limit.
    segment = _normalize_segments((segment,))[0]
    if frame_paths is not None:
        paths = list(frame_paths)
    else:
        source_root = Path(frames_folder)
        segment_root = source_root / segment
        paths = _frame_paths(segment_root if segment_root.is_dir() else source_root)
    rng = np.random.default_rng(random_state)
    # Shuffle only for a soft limit; a full compile keeps deterministic order.
    if limit is not None:
        paths = [paths[index] for index in rng.permutation(len(paths))]
    batch_size = (
        max(1, int(scan_batch_size)) if scan_batch_size is not None
            else 512 if limit is None else min(256, max(32, int(limit) * 2))
    )
    collected = []
    replay_ids: set[str] = set()
    total_batches = (len(paths) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(paths), batch_size), start=1):
        batch_paths = paths[start:start + batch_size]
        try:
            batch = _scan_frame_paths(
                batch_paths,
                (segment,),
                include_event_context,
                selected_replay_ids,
            ).collect(engine='streaming')
        except ValueError as error:
            if 'No event frame rows matched' not in str(error):
                raise
            batch = pl.DataFrame()
        if not batch.is_empty():
            # Retain only this segment before opening the next bounded path group.
            collected.append(batch)
            replay_ids.update(batch['replay_id'].drop_nulls().unique().to_list())
        target = 'all' if limit is None else str(limit)
        print(
            f'[{segment}] scan {batch_number}/{total_batches}: '
            f'{len(replay_ids)}/{target} replays collected',
            flush=True,
        )
        if limit is not None and len(replay_ids) >= int(limit):
            break
    if not collected:
        return pl.DataFrame()
    # Parser versions and pass availability can leave optional columns absent by batch.
    frames = pl.concat(collected, how='diagonal_relaxed', rechunk=False)
    return limit_replays(frames, limit, random_state)


def limit_replays(
    frames: pl.DataFrame,
    limit: int | None,
    random_state: int = RANDOM_STATE,
) -> pl.DataFrame:
    # Select all shot rows from at most ``limit`` randomly chosen replays.
    if limit is None:
        return frames
    limit = int(limit)
    if limit <= 0:
        raise ValueError('limit must be a positive integer when provided')
    replay_ids = frames['replay_id'].drop_nulls().unique().sort().to_numpy()
    if limit >= len(replay_ids):
        return frames
    rng = np.random.default_rng(random_state)
    selected = rng.choice(replay_ids, size=limit, replace=False)
    return frames.filter(pl.col('replay_id').is_in(selected))


def _float_array(frame: pl.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(frame.height, default, dtype=np.float32)
    return np.asarray(frame[column].cast(pl.Float32, strict=False).to_numpy(), dtype=np.float32)


def _string_array(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(frame.height, '', dtype=str)
    return np.asarray(frame[column].cast(pl.String, strict=False).fill_null('').to_numpy(), dtype=str)


def _bool_array(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(frame.height, dtype=bool)
    return np.asarray(frame[column].cast(pl.Boolean, strict=False).fill_null(False).to_numpy(), dtype=bool)


def _safe_speed(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(x) + np.square(y) + np.square(z))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=np.isfinite(denominator) & (np.abs(denominator) > 1e-6),
    )


def _vector_angle(
    left_x: np.ndarray,
    left_y: np.ndarray,
    left_z: np.ndarray,
    right_x: np.ndarray,
    right_y: np.ndarray,
    right_z: np.ndarray,
) -> np.ndarray:
    denominator = _safe_speed(left_x, left_y, left_z) * _safe_speed(
        right_x,
        right_y,
        right_z,
    )
    cosine = _safe_divide(
        left_x * right_x + left_y * right_y + left_z * right_z,
        denominator,
    )
    angle = np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32)
    angle[~np.isfinite(denominator) | (denominator <= 1e-6)] = np.nan
    return angle


def _planar_angle(
    left_a: np.ndarray,
    left_b: np.ndarray,
    right_a: np.ndarray,
    right_b: np.ndarray,
) -> np.ndarray:
    cross = left_a * right_b - left_b * right_a
    dot = left_a * right_a + left_b * right_b
    angle = np.abs(np.arctan2(cross, dot)).astype(np.float32)
    valid = (
        np.isfinite(left_a) & np.isfinite(left_b)
        & np.isfinite(right_a) & np.isfinite(right_b)
        & ((np.abs(left_a) + np.abs(left_b)) > 1e-6)
        & ((np.abs(right_a) + np.abs(right_b)) > 1e-6)
    )
    angle[~valid] = np.nan
    return angle


def _forward_vector(yaw: np.ndarray, pitch: np.ndarray) -> tuple[np.ndarray, ...]:
    horizontal = np.cos(pitch)
    return horizontal * np.cos(yaw), horizontal * np.sin(yaw), np.sin(pitch)


def _role_spatial_features(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    own_net_y: float,
) -> dict[str, np.ndarray]:
    forward_x, forward_y, forward_z = _forward_vector(yaw, pitch)
    ball_relative = (ball_x - pos_x, ball_y - pos_y, ball_z - pos_z)
    own_net_relative = (-pos_x, own_net_y - pos_y, GOAL_HEIGHT / 2.0 - pos_z)
    opp_net_relative = (-pos_x, -own_net_y - pos_y, GOAL_HEIGHT / 2.0 - pos_z)
    return {
        'distance_to_ball': _safe_speed(*ball_relative),
        'angle_to_ball': _vector_angle(
            forward_x, forward_y, forward_z, *ball_relative,
        ),
        'distance_to_own_net': _safe_speed(*own_net_relative),
        'angle_to_own_net': _vector_angle(
            forward_x, forward_y, forward_z, *own_net_relative,
        ),
        'distance_to_opp_net': _safe_speed(*opp_net_relative),
        'angle_to_opp_net': _vector_angle(
            forward_x, forward_y, forward_z, *opp_net_relative,
        ),
        'angle_front_car_to_ground': np.abs(
            np.arctan2(forward_z, np.hypot(forward_x, forward_y))
        ).astype(np.float32),
    }


def _contact_features(
    player_x: np.ndarray,
    player_y: np.ndarray,
    player_z: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    ball_start_x: np.ndarray,
    ball_start_y: np.ndarray,
    ball_start_z: np.ndarray,
    ball_end_x: np.ndarray,
    ball_end_y: np.ndarray,
    ball_end_z: np.ndarray,
) -> dict[str, np.ndarray]:
    player_to_ball = (
        ball_start_x - player_x,
        ball_start_y - player_y,
        ball_start_z - player_z,
    )
    distance = _safe_speed(*player_to_ball)
    contact_x = ball_start_x - _safe_divide(player_to_ball[0], distance) * BALL_RADIUS
    contact_y = ball_start_y - _safe_divide(player_to_ball[1], distance) * BALL_RADIUS
    contact_z = ball_start_z - _safe_divide(player_to_ball[2], distance) * BALL_RADIUS
    forward_x, forward_y, forward_z = _forward_vector(yaw, pitch)
    start_relative = (
        ball_start_x - contact_x,
        ball_start_y - contact_y,
        ball_start_z - contact_z,
    )
    end_relative = (
        ball_end_x - contact_x,
        ball_end_y - contact_y,
        ball_end_z - contact_z,
    )
    return {
        'contact_x': contact_x,
        'contact_y': contact_y,
        'contact_z': contact_z,
        'angle_contact_to_ball_start_xy': _planar_angle(
            forward_x, forward_y, start_relative[0], start_relative[1],
        ),
        'angle_contact_to_ball_start_xz': _planar_angle(
            forward_x, forward_z, start_relative[0], start_relative[2],
        ),
        'angle_contact_to_ball_end_xy': _planar_angle(
            forward_x, forward_y, end_relative[0], end_relative[1],
        ),
        'angle_contact_to_ball_end_xz': _planar_angle(
            forward_x, forward_z, end_relative[0], end_relative[2],
        ),
    }


def _ball_at_time(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    ball_vx: np.ndarray,
    ball_vy: np.ndarray,
    ball_vz: np.ndarray,
    seconds: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return (
        ball_x + ball_vx * seconds,
        ball_y + ball_vy * seconds,
        np.maximum(ball_z + ball_vz * seconds - 0.5 * GRAVITY * np.square(seconds), 0.0),
    )


def _shot_path(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    ball_vx: np.ndarray,
    ball_vy: np.ndarray,
    ball_vz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crossing_time = _safe_divide(GOAL_PLANE_Y - ball_y, ball_vy)
    valid = (
        np.isfinite(crossing_time)
        & (crossing_time > 0.0)
        & (crossing_time <= TRAJECTORY_SECONDS)
    )
    end_time = np.where(valid, crossing_time, TRAJECTORY_SECONDS).astype(np.float32)
    end_x, end_y, end_z = _ball_at_time(
        ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz, end_time,
    )
    return end_time, end_x, end_y, end_z, valid


def _trajectory_intercept(
    player_x: np.ndarray,
    player_y: np.ndarray,
    player_z: np.ndarray,
    player_speed: np.ndarray,
    player_boost: np.ndarray,
    player_vel_x: np.ndarray,
    player_vel_y: np.ndarray,
    player_vel_z: np.ndarray,
    player_yaw: np.ndarray,
    player_on_ground: np.ndarray,
    player_has_dodge: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    ball_vx: np.ndarray,
    ball_vy: np.ndarray,
    ball_vz: np.ndarray,
    horizon: np.ndarray,
    valid_path: np.ndarray,
) -> dict[str, np.ndarray]:
    row_count = len(player_x)
    min_distance = np.full(row_count, np.nan, dtype=np.float32)
    min_arrival = np.full(row_count, np.nan, dtype=np.float32)
    intercept_time = np.full(row_count, np.inf, dtype=np.float32)
    intercept_margin = np.full(row_count, np.nan, dtype=np.float32)
    intercept_x = np.full(row_count, np.nan, dtype=np.float32)
    intercept_y = np.full(row_count, np.nan, dtype=np.float32)
    intercept_z = np.full(row_count, np.nan, dtype=np.float32)
    intercept_distance = np.full(row_count, np.nan, dtype=np.float32)
    boost = np.clip(np.nan_to_num(player_boost, nan=0.0), 0.0, 100.0)
    on_ground = np.asarray(player_on_ground, dtype=bool)
    has_dodge = np.asarray(player_has_dodge, dtype=bool)
    active = valid_path & np.isfinite(player_speed) & np.isfinite(player_yaw)
    boost_needed_at_intercept = np.full(row_count, np.nan, dtype=np.float32)
    for sample in range(1, TRAJECTORY_SAMPLES + 1):
        seconds = horizon * (sample / TRAJECTORY_SAMPLES)
        target_x, target_y, target_z = _ball_at_time(
            ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz, seconds,
        )
        delta_x = target_x - player_x
        delta_y = target_y - player_y
        delta_z = target_z - player_z
        distance = _safe_speed(delta_x, delta_y, delta_z)
        planar_distance = np.hypot(delta_x, delta_y)
        direction_x = _safe_divide(delta_x, np.maximum(planar_distance, 1e-6))
        direction_y = _safe_divide(delta_y, np.maximum(planar_distance, 1e-6))
        speed_toward = np.maximum(
            player_vel_x * direction_x + player_vel_y * direction_y, 0.0
        )

        # Approximate the delay to point the car toward the target using the
        # measured ~1.55 s half-turn time. The model then grants bounded
        # throttle/boost acceleration and respects the boost fuel budget.
        desired_yaw = np.arctan2(delta_y, delta_x)
        yaw_delta = np.arctan2(
            np.sin(desired_yaw - player_yaw),
            np.cos(desired_yaw - player_yaw),
        )
        turn_time = np.abs(yaw_delta) * (1.55 / np.pi)
        drive_time = np.maximum(seconds - turn_time, 0.0)
        throttle_accel = np.where(
            on_ground,
            CAR_GROUND_THROTTLE_ACCEL
            * np.clip(1.0 - speed_toward / CAR_GROUND_SPEED_CAP, 0.0, 1.0),
            CAR_AIR_THROTTLE_ACCEL,
        )
        boost_accel = np.where(on_ground, BOOST_ACCEL_GROUND, BOOST_ACCEL_AIR)
        boost_time = np.minimum(
            drive_time,
            boost / BOOST_CONSUMPTION_PER_SECOND,
        )
        base_travel = (
            speed_toward * drive_time
            + 0.5 * throttle_accel * np.square(drive_time)
        )
        boost_travel = boost_accel * (
            boost_time * drive_time - 0.5 * np.square(boost_time)
        )
        reachable_distance = np.minimum(
            base_travel + boost_travel,
            CAR_MAX_SPEED * drive_time,
        )

        # Vertical reach uses Rocket League's jump impulse/hold bonus and an
        # optimistic aerial boost envelope. This is a capability feature, not
        # a full control simulation; collisions and exact car orientation in
        # flight are not modeled.
        ballistic_z = player_vel_z * seconds - 0.5 * GRAVITY * np.square(seconds)
        jump_impulse = np.where(
            on_ground,
            292.0 + 292.0 * np.clip(seconds / 0.2, 0.0, 1.0)
            + np.where(has_dodge, 292.0, 0.0),
            np.where(has_dodge, 292.0, 0.0),
        )
        aerial_boost_lift = 0.5 * BOOST_ACCEL_AIR * np.square(boost_time)
        vertical_error = delta_z - ballistic_z
        vertical_reachable = (
            (vertical_error <= jump_impulse + aerial_boost_lift + BALL_RADIUS)
            & (vertical_error >= -BALL_RADIUS)
        )
        arrival_speed = np.minimum(
            speed_toward + throttle_accel * drive_time + boost_accel * boost_time,
            CAR_MAX_SPEED,
        )
        arrival = turn_time + _safe_divide(
            planar_distance,
            np.maximum(arrival_speed, CAR_MIN_INTERCEPT_SPEED),
        )
        sample_valid = active & np.isfinite(distance) & np.isfinite(arrival)
        min_distance = np.where(
            sample_valid & (~np.isfinite(min_distance) | (distance < min_distance)),
            distance,
            min_distance,
        )
        min_arrival = np.where(
            sample_valid & (~np.isfinite(min_arrival) | (arrival < min_arrival)),
            arrival,
            min_arrival,
        )
        feasible = (
            sample_valid
            & (seconds >= turn_time)
            & (reachable_distance >= planar_distance)
            & vertical_reachable
            & (seconds < intercept_time)
        )
        intercept_time = np.where(feasible, seconds, intercept_time)
        intercept_margin = np.where(feasible, seconds - arrival, intercept_margin)
        intercept_x = np.where(feasible, target_x, intercept_x)
        intercept_y = np.where(feasible, target_y, intercept_y)
        intercept_z = np.where(feasible, target_z, intercept_z)
        intercept_distance = np.where(feasible, distance, intercept_distance)
        baseline_travel = base_travel
        extra_distance = np.maximum(planar_distance - baseline_travel, 0.0)
        discriminant = np.maximum(
            np.square(drive_time) - 2.0 * extra_distance / np.maximum(boost_accel, 1e-6),
            0.0,
        )
        required_boost_time = np.where(
            extra_distance > 0.0,
            drive_time - np.sqrt(discriminant),
            0.0,
        )
        required_boost = required_boost_time * BOOST_CONSUMPTION_PER_SECOND
        boost_needed_at_intercept = np.where(
            feasible & (required_boost <= boost),
            required_boost,
            boost_needed_at_intercept,
        )
    possible = np.isfinite(intercept_time)
    boost_needed_at_intercept[~possible] = np.nan
    return {
        'min_distance': min_distance,
        'min_arrival': min_arrival,
        'possible': possible,
        'requires_aerial': possible & (intercept_z > player_z + AERIAL_INTERCEPT_Z),
        'margin': intercept_margin,
        'x': intercept_x,
        'y': intercept_y,
        'z': intercept_z,
        'boost_needed': boost_needed_at_intercept,
        'boost_surplus': np.where(possible, boost - boost_needed_at_intercept, np.nan),
    }


def _opponent_event_features(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
    vel_x: np.ndarray,
    vel_y: np.ndarray,
    vel_z: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    boost: np.ndarray,
    on_ground: np.ndarray,
    has_dodge: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    ball_vx: np.ndarray,
    ball_vy: np.ndarray,
    ball_vz: np.ndarray,
    ball_end_x: np.ndarray,
    ball_end_y: np.ndarray,
    ball_end_z: np.ndarray,
    shot_path_time: np.ndarray,
    shot_path_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    player_speed = _safe_speed(vel_x, vel_y, vel_z)
    forward_x, forward_y, forward_z = _forward_vector(yaw, pitch)
    ball_start_relative = (ball_x - pos_x, ball_y - pos_y, ball_z - pos_z)
    ball_end_relative = (
        ball_end_x - pos_x,
        ball_end_y - pos_y,
        ball_end_z - pos_z,
    )
    defensive_net_relative = (
        -pos_x,
        GOAL_PLANE_Y - pos_y,
        GOAL_HEIGHT / 2.0 - pos_z,
    )
    # These point-to-point values are constant-speed baselines; intercept
    # features below use the acceleration/turn/boost-budget reachability model.
    effective_speed = np.maximum(player_speed, CAR_MIN_INTERCEPT_SPEED)
    trajectory = _trajectory_intercept(
        pos_x,
        pos_y,
        pos_z,
        player_speed,
        boost,
        vel_x,
        vel_y,
        vel_z,
        yaw,
        on_ground,
        has_dodge,
        ball_x,
        ball_y,
        ball_z,
        ball_vx,
        ball_vy,
        ball_vz,
        np.full(len(pos_x), TRAJECTORY_SECONDS, dtype=np.float32),
        np.isfinite(pos_x),
    )
    shot_path = _trajectory_intercept(
        pos_x,
        pos_y,
        pos_z,
        player_speed,
        boost,
        vel_x,
        vel_y,
        vel_z,
        yaw,
        on_ground,
        has_dodge,
        ball_x,
        ball_y,
        ball_z,
        ball_vx,
        ball_vy,
        ball_vz,
        shot_path_time,
        shot_path_valid,
    )
    return {
        'proximity_to_ball_start_x': ball_start_relative[0],
        'proximity_to_ball_start_y': ball_start_relative[1],
        'proximity_to_ball_start_z': ball_start_relative[2],
        'proximity_to_ball_end_x': ball_end_relative[0],
        'proximity_to_ball_end_y': ball_end_relative[1],
        'proximity_to_ball_end_z': ball_end_relative[2],
        'proximity_to_defensive_net_location_x': defensive_net_relative[0],
        'proximity_to_defensive_net_location_y': defensive_net_relative[1],
        'proximity_to_defensive_net_location_z': defensive_net_relative[2],
        'angle_proximity_to_ball_start_xy': _planar_angle(
            forward_x, forward_y, ball_start_relative[0], ball_start_relative[1],
        ),
        'angle_proximity_to_ball_start_xz': _planar_angle(
            forward_x, forward_z, ball_start_relative[0], ball_start_relative[2],
        ),
        'angle_proximity_to_ball_end_xy': _planar_angle(
            forward_x, forward_y, ball_end_relative[0], ball_end_relative[1],
        ),
        'angle_proximity_to_ball_end_xz': _planar_angle(
            forward_x, forward_z, ball_end_relative[0], ball_end_relative[2],
        ),
        'angle_proximity_to_defensive_net_to_defensive_net_xy': _planar_angle(
            forward_x,
            forward_y,
            defensive_net_relative[0],
            defensive_net_relative[1],
        ),
        'angle_proximity_to_defensive_net_to_defensive_net_xz': _planar_angle(
            forward_x,
            forward_z,
            defensive_net_relative[0],
            defensive_net_relative[2],
        ),
        'time_to_ball_end': _safe_divide(
            _safe_speed(*ball_end_relative),
            np.maximum(player_speed, CAR_MIN_INTERCEPT_SPEED),
        ),
        'min_time_to_ball_trajectory': trajectory['min_arrival'],
        'intercept_time_margin': trajectory['margin'],
        'intercept_x': trajectory['x'],
        'intercept_y': trajectory['y'],
        'intercept_z': trajectory['z'],
        'min_distance_to_trajectory': trajectory['min_distance'],
        'boost_needed_to_intercept': trajectory['boost_needed'],
        'boost_surplus_at_intercept': trajectory['boost_surplus'],
        'time_to_ball_start_estimate': _safe_divide(
            _safe_speed(*ball_start_relative),
            effective_speed,
        ),
        'time_to_ball_end_estimate': _safe_divide(
            _safe_speed(*ball_end_relative),
            effective_speed,
        ),
        'time_to_defensive_net_estimate': _safe_divide(
            _safe_speed(*defensive_net_relative),
            effective_speed,
        ),
        'time_to_shot_path_estimate': shot_path['min_arrival'],
        'shot_path_intercept_time_margin': shot_path['margin'],
        'shot_path_intercept_x': shot_path['x'],
        'shot_path_intercept_y': shot_path['y'],
        'shot_path_intercept_z': shot_path['z'],
        'min_distance_to_shot_path': shot_path['min_distance'],
        'intercept_possible': trajectory['possible'],
        'intercept_requires_aerial': trajectory['requires_aerial'],
        'shot_path_intercept_possible': shot_path['possible'],
    }


def _shooter_values(frames: pl.DataFrame, field: str) -> np.ndarray:
    slots = _float_array(frames, 'event_player_1_slot', 0).astype(np.int8)
    return _selected_player_values(frames, field, slots)


def _selected_player_values(
    frames: pl.DataFrame,
    field: str,
    slots: np.ndarray,
) -> np.ndarray:
    output = np.full(frames.height, np.nan, dtype=np.float32)
    for player_number in range(1, MAX_PLAYERS + 1):
        mask = slots == player_number
        if np.any(mask):
            output[mask] = _float_array(frames, f'player_{player_number}_{field}')[mask]
    return output


def _player_catalog_features(
    value: Callable[[str], np.ndarray],
    direction: np.ndarray,
    fields: Sequence[str],
) -> dict[str, np.ndarray]:
    features = {}
    for field in fields:
        if field in PLAYER_BASE_MODEL_FIELDS or field in PLAYER_QUATERNION_FIELDS:
            continue
        values = value(field)
        if field in PLAYER_WORLD_XY_FIELDS:
            values = values * direction
        elif field == 'yaw':
            values = (
                values
                + np.where(direction < 0.0, np.pi, 0.0)
                + np.pi
            ) % (2.0 * np.pi) - np.pi
        elif field == 'camera_frame_yaw':
            values = (
                values
                + np.where(direction < 0.0, 32768.0, 0.0)
                + 32768.0
            ) % 65536.0 - 32768.0
        features[PLAYER_FEATURE_ALIASES.get(field, field)] = values

    # Rotate orange quaternions by 180 degrees around world Z with the other geometry.
    rot_x, rot_y, rot_z, rot_w = (
        value(field).copy()
        for field in PLAYER_QUATERNION_FIELDS
    )
    orange = direction < 0.0
    normalized = {
        'rotation_x': np.where(orange, -rot_y, rot_x),
        'rotation_y': np.where(orange, rot_x, rot_y),
        'rotation_z': np.where(orange, rot_w, rot_z),
        'rotation_w': np.where(orange, -rot_z, rot_w),
    }
    features.update(normalized)
    return features


def prepare_model_data(
    frames: pl.DataFrame,
    include_prior_event_history: bool = False,
) -> pl.DataFrame:
    # Engineer attack-normalized ball, shooter, and nearest-defender features.
    shooter_team = np.char.lower(_string_array(frames, 'shooter_team'))
    # Orient every possession toward positive Y so both teams share one field geometry.
    direction = np.where(shooter_team == 'orange', -1.0, 1.0).astype(np.float32)
    ball_x_raw = _float_array(frames, 'ball_pos_x')
    ball_y_raw = _float_array(frames, 'ball_pos_y')
    ball_z = _float_array(frames, 'ball_pos_z')
    ball_vx_raw = _float_array(frames, 'ball_vel_x')
    ball_vy_raw = _float_array(frames, 'ball_vel_y')
    ball_vz = _float_array(frames, 'ball_vel_z')
    ball_x = ball_x_raw * direction
    ball_y = ball_y_raw * direction
    ball_vx = ball_vx_raw * direction
    ball_vy = ball_vy_raw * direction
    ball_speed = _safe_speed(ball_vx, ball_vy, ball_vz)
    goal_dx = -ball_x
    goal_dy = GOAL_PLANE_Y - ball_y
    goal_distance = np.sqrt(np.square(goal_dx) + np.square(goal_dy))
    time_to_goal = _safe_divide(goal_dy, ball_vy)
    projected_x = ball_x + ball_vx * time_to_goal
    projected_z = ball_z + ball_vz * time_to_goal - 325.0 * np.square(time_to_goal)
    shot_path_time, shot_end_x, shot_end_y, shot_end_z, shot_path_valid = _shot_path(
        ball_x,
        ball_y,
        ball_z,
        ball_vx,
        ball_vy,
        ball_vz,
    )

    shooter = {
        field: _shooter_values(frames, field)
        for field in ('pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'boost_raw')
    }
    shooter_x = shooter['pos_x'] * direction
    shooter_y = shooter['pos_y'] * direction
    shooter_vx = shooter['vel_x'] * direction
    shooter_vy = shooter['vel_y'] * direction
    shooter_speed = _safe_speed(shooter_vx, shooter_vy, shooter['vel_z'])
    shooter_ball_distance = _safe_speed(ball_x - shooter_x, ball_y - shooter_y, ball_z - shooter['pos_z'])
    shooter_catalog = _player_catalog_features(
        lambda field: _shooter_values(frames, field),
        direction,
        SHOOTER_FIELDS,
    )
    shooter_spatial = _role_spatial_features(
        shooter_x,
        shooter_y,
        shooter['pos_z'],
        shooter_catalog['yaw'],
        shooter_catalog['pitch'],
        ball_x,
        ball_y,
        ball_z,
        -GOAL_PLANE_Y,
    )

    data: dict[str, Any] = {
        'replay_id': _string_array(frames, 'replay_id'),
        'event_id': _string_array(frames, 'event_id'),
        'is_goal': _float_array(frames, 'is_goal', 0).astype(np.int8),
        'ball_x': ball_x,
        'ball_y': ball_y,
        'ball_z': ball_z,
        'ball_velocity_x': ball_vx,
        'ball_velocity_y': ball_vy,
        'ball_velocity_z': ball_vz,
        'ball_speed': ball_speed,
        'ball_goal_distance': goal_distance,
        'ball_goal_angle': np.arctan2(np.abs(goal_dx), np.maximum(goal_dy, 1.0)),
        'ball_goal_elevation': np.arctan2(ball_z, np.maximum(goal_distance, 1.0)),
        'ball_projected_x': projected_x,
        'ball_projected_z': projected_z,
        'ball_projected_inside_goal': (
            (time_to_goal > 0.0)
            & (np.abs(projected_x) <= GOAL_CENTER_TO_POST - BALL_RADIUS)
            & (projected_z >= BALL_RADIUS)
            & (projected_z <= GOAL_HEIGHT - BALL_RADIUS)
        ),
        'shooter_x': shooter_x,
        'shooter_y': shooter_y,
        'shooter_z': shooter['pos_z'],
        'shooter_velocity_x': shooter_vx,
        'shooter_velocity_y': shooter_vy,
        'shooter_velocity_z': shooter['vel_z'],
        'shooter_speed': shooter_speed,
        'shooter_boost': shooter['boost_raw'],
        'shooter_ball_relative_x': ball_x - shooter_x,
        'shooter_ball_relative_y': ball_y - shooter_y,
        'shooter_ball_relative_z': ball_z - shooter['pos_z'],
        'shooter_time_to_ball': _safe_divide(
            shooter_ball_distance,
            np.maximum(shooter_speed, 500.0),
        ),
    }
    shooter_on_field = _shooter_values(frames, 'on_field')
    data['shooter_on_field'] = np.isfinite(shooter_on_field) & (shooter_on_field != 0.0)
    data.update({f'shooter_{field}': values for field, values in shooter_catalog.items()})
    data.update({f'shooter_{field}': values for field, values in shooter_spatial.items()})
    data.update({
        f'shooter_{field}': values
        for field, values in _contact_features(
            shooter_x,
            shooter_y,
            shooter['pos_z'],
            shooter_catalog['yaw'],
            shooter_catalog['pitch'],
            ball_x,
            ball_y,
            ball_z,
            shot_end_x,
            shot_end_y,
            shot_end_z,
        ).items()
    })
    if include_prior_event_history and 'history_seconds_since_kickoff' in frames.columns:
        data['shooter_team_has_possession'] = _bool_array(frames, 'shooter_team_has_possession')
        data['shooter_team_possession_length'] = _float_array(
            frames,
            'shooter_team_possession_length',
            0.0,
        )
        pass_seconds = _float_array(frames, '_pass_seconds')
        shot_seconds = _float_array(frames, 'frame_seconds_elapsed')
        pass_ball_x = _float_array(frames, '_pass_ball_pos_x') * direction
        pass_ball_y = _float_array(frames, '_pass_ball_pos_y') * direction
        pass_ball_z = _float_array(frames, '_pass_ball_pos_z')
        pass_ball_vx = _float_array(frames, '_pass_ball_vel_x') * direction
        pass_ball_vy = _float_array(frames, '_pass_ball_vel_y') * direction
        pass_ball_vz = _float_array(frames, '_pass_ball_vel_z')
        pass_ball_speed = _safe_speed(pass_ball_vx, pass_ball_vy, pass_ball_vz)
        passer_x = _float_array(frames, '_passer_pos_x') * direction
        passer_y = _float_array(frames, '_passer_pos_y') * direction
        passer_z = _float_array(frames, '_passer_pos_z')
        passer_vx = _float_array(frames, '_passer_vel_x') * direction
        passer_vy = _float_array(frames, '_passer_vel_y') * direction
        passer_vz = _float_array(frames, '_passer_vel_z')
        passer_speed = _safe_speed(passer_vx, passer_vy, passer_vz)
        passer_ball_distance = _safe_speed(
            pass_ball_x - passer_x,
            pass_ball_y - passer_y,
            pass_ball_z - passer_z,
        )
        passer_catalog = _player_catalog_features(
            lambda field: _float_array(frames, f'_passer_{field}'),
            direction,
            PASSER_FIELDS,
        )
        passer_spatial = _role_spatial_features(
            passer_x,
            passer_y,
            passer_z,
            passer_catalog['yaw'],
            passer_catalog['pitch'],
            pass_ball_x,
            pass_ball_y,
            pass_ball_z,
            -GOAL_PLANE_Y,
        )
        pass_displacement_x = ball_x - pass_ball_x
        pass_displacement_y = ball_y - pass_ball_y
        pass_displacement_z = ball_z - pass_ball_z
        data.update({
            'pass_seconds_to_shot': np.maximum(shot_seconds - pass_seconds, 0.0),
            'pass_ball_x': pass_ball_x,
            'pass_ball_y': pass_ball_y,
            'pass_ball_z': pass_ball_z,
            'pass_ball_velocity_x': pass_ball_vx,
            'pass_ball_velocity_y': pass_ball_vy,
            'pass_ball_velocity_z': pass_ball_vz,
            'pass_ball_speed': pass_ball_speed,
            'pass_ball_displacement_x': pass_displacement_x,
            'pass_ball_displacement_y': pass_displacement_y,
            'pass_ball_displacement_z': pass_displacement_z,
            'pass_ball_distance': _safe_speed(
                pass_displacement_x,
                pass_displacement_y,
                pass_displacement_z,
            ),
            'pass_ball_velocity_change_x': ball_vx - pass_ball_vx,
            'pass_ball_velocity_change_y': ball_vy - pass_ball_vy,
            'pass_ball_velocity_change_z': ball_vz - pass_ball_vz,
            'pass_ball_speed_change': ball_speed - pass_ball_speed,
            'passer_on_field': _bool_array(frames, '_passer_on_field'),
            'passer_x': passer_x,
            'passer_y': passer_y,
            'passer_z': passer_z,
            'passer_velocity_x': passer_vx,
            'passer_velocity_y': passer_vy,
            'passer_velocity_z': passer_vz,
            'passer_speed': passer_speed,
            'passer_boost': _float_array(frames, '_passer_boost_raw'),
            'passer_ball_relative_x': pass_ball_x - passer_x,
            'passer_ball_relative_y': pass_ball_y - passer_y,
            'passer_ball_relative_z': pass_ball_z - passer_z,
            'passer_time_to_ball': _safe_divide(
                passer_ball_distance,
                np.maximum(passer_speed, 500.0),
            ),
        })
        data.update({f'passer_{field}': values for field, values in passer_catalog.items()})
        data.update({f'passer_{field}': values for field, values in passer_spatial.items()})
        data.update({
            f'passer_{field}': values
            for field, values in _contact_features(
                passer_x,
                passer_y,
                passer_z,
                passer_catalog['yaw'],
                passer_catalog['pitch'],
                pass_ball_x,
                pass_ball_y,
                pass_ball_z,
                ball_x,
                ball_y,
                ball_z,
            ).items()
        })
        for column in OFF_COLUMNS:
            data[column] = _bool_array(frames, column)
        data['history_seconds_since_kickoff'] = _float_array(frames, 'history_seconds_since_kickoff')
        for event_type in HISTORY_EVENT_TYPES:
            column = f'history_weighted_{event_type}_count'
            data[column] = _float_array(frames, column, 0.0)

    # Distance ordering gives defenders stable feature slots across every game mode.
    defender_distances = np.full(
        (frames.height, MAX_PLAYERS),
        np.inf,
        dtype=np.float32,
    )
    for player_number in range(1, MAX_PLAYERS + 1):
        team = np.char.lower(_string_array(frames, f'player_{player_number}_team'))
        on_field = _float_array(frames, f'player_{player_number}_on_field', 0) != 0
        pos_x = _float_array(frames, f'player_{player_number}_pos_x') * direction
        pos_y = _float_array(frames, f'player_{player_number}_pos_y') * direction
        pos_z = _float_array(frames, f'player_{player_number}_pos_z')
        distance = _safe_speed(pos_x - ball_x, pos_y - ball_y, pos_z - ball_z)
        eligible = on_field & (team != '') & (team != shooter_team) & np.isfinite(distance)
        defender_distances[:, player_number - 1] = np.where(eligible, distance, np.inf)

    defender_order = np.argsort(defender_distances, axis=1)[:, :4]
    ordered_distances = np.take_along_axis(defender_distances, defender_order, axis=1)
    defender_slots = (defender_order + 1).astype(np.int8)
    defender_slots[~np.isfinite(ordered_distances)] = 0
    del defender_distances, defender_order

    for defender_number in range(1, 5):
        slots = defender_slots[:, defender_number - 1]
        distance = ordered_distances[:, defender_number - 1]
        distance = np.where(np.isfinite(distance), distance, np.nan).astype(np.float32)
        pos_x = _selected_player_values(frames, 'pos_x', slots) * direction
        pos_y = _selected_player_values(frames, 'pos_y', slots) * direction
        pos_z = _selected_player_values(frames, 'pos_z', slots)
        vel_x = _selected_player_values(frames, 'vel_x', slots) * direction
        vel_y = _selected_player_values(frames, 'vel_y', slots) * direction
        vel_z = _selected_player_values(frames, 'vel_z', slots)
        boost = _selected_player_values(frames, 'boost_raw', slots)
        speed = _safe_speed(vel_x, vel_y, vel_z)
        prefix = f'defender_{defender_number}'
        defender_catalog = _player_catalog_features(
            lambda field, slots=slots: _selected_player_values(frames, field, slots),
            direction,
            DEFENDER_FIELDS,
        )
        defender_spatial = _role_spatial_features(
            pos_x,
            pos_y,
            pos_z,
            defender_catalog['yaw'],
            defender_catalog['pitch'],
            ball_x,
            ball_y,
            ball_z,
            GOAL_PLANE_Y,
        )
        data.update({
            f'{prefix}_on_field': slots != 0,
            f'{prefix}_x': pos_x,
            f'{prefix}_y': pos_y,
            f'{prefix}_z': pos_z,
            f'{prefix}_velocity_x': vel_x,
            f'{prefix}_velocity_y': vel_y,
            f'{prefix}_velocity_z': vel_z,
            f'{prefix}_speed': speed,
            f'{prefix}_boost': boost,
            f'{prefix}_time_to_ball': _safe_divide(distance, np.maximum(speed, 500.0)),
            f'{prefix}_ball_relative_x': ball_x - pos_x,
            f'{prefix}_ball_relative_y': ball_y - pos_y,
            f'{prefix}_ball_relative_z': ball_z - pos_z,
        })
        data.update({
            f'{prefix}_{field}': values
            for field, values in defender_catalog.items()
        })
        data.update({
            f'{prefix}_{field}': values
            for field, values in defender_spatial.items()
        })
        data.update({
            f'{prefix}_{field}': values
            for field, values in _opponent_event_features(
                pos_x,
                pos_y,
                pos_z,
                vel_x,
                vel_y,
                vel_z,
                defender_catalog['yaw'],
                defender_catalog['pitch'],
                boost,
                defender_catalog['on_ground'],
                defender_catalog['has_dodge'],
                ball_x,
                ball_y,
                ball_z,
                ball_vx,
                ball_vy,
                ball_vz,
                shot_end_x,
                shot_end_y,
                shot_end_z,
                shot_path_time,
                shot_path_valid,
            ).items()
        })
    return pl.DataFrame(data)


def to_sparse_matrix(
    features: pl.DataFrame,
    feature_columns: Sequence[str],
) -> sparse.csr_matrix:
    # Build CSR one feature at a time without a full dense intermediate.
    # Column-wise construction avoids materializing the full rows-by-features array.
    columns = []
    for column in feature_columns:
        values = np.array(
            features[column].cast(pl.Float32, strict=False).to_numpy(),
            dtype=np.float32,
            copy=True,
        )
        values[~np.isfinite(values)] = np.nan
        columns.append(sparse.csr_matrix(values.reshape(-1, 1)))
    if not columns:
        return sparse.csr_matrix((features.height, 0), dtype=np.float32)
    matrix = sparse.hstack(columns, format='csr', dtype=np.float32)
    del columns
    return matrix


def _feature_matrix(
    features: pl.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    if features.height == 0:
        return np.empty((0, len(feature_columns)), dtype=np.float32)
    columns = []
    for column in feature_columns:
        values = np.array(
            features[column].cast(pl.Float32, strict=False).to_numpy(),
            dtype=np.float32,
            copy=True,
        )
        values[~np.isfinite(values)] = np.nan
        columns.append(values)
    return np.column_stack(columns).astype(np.float32, copy=False)


def _variable_feature_columns(
    features: pl.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[list[str], list[str]]:
    # Select feature classes from training rows only so held-out replays remain unseen.
    class_counts = features.select([
        pl.col(column)
        .cast(pl.Float64, strict=False)
        .filter(pl.col(column).cast(pl.Float64, strict=False).is_finite())
        .n_unique()
        .alias(column)
        for column in feature_columns
    ]).row(0, named=True)
    selected = [
        column
        for column in feature_columns
        if int(class_counts.get(column) or 0) > 1
    ]
    removed = [column for column in feature_columns if column not in selected]
    if not selected:
        raise ValueError('Training data does not contain any variable model features')
    return selected, removed


def _feature_column_class(column: str) -> str:
    if column.startswith('shooter_'):
        return 'shooter'
    if column.startswith('passer_'):
        return 'passer'
    if column.startswith('defender_'):
        return 'defender'
    if column.startswith('pass_'):
        return 'pass'
    if column.startswith('ball_'):
        return 'ball'
    if column.startswith('off_'):
        return 'offensive_context'
    if column.startswith('history_'):
        return 'history'
    return 'context'


def _feature_column_classes(feature_columns: Sequence[str]) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {}
    for column in feature_columns:
        classes.setdefault(_feature_column_class(column), []).append(column)
    return classes


