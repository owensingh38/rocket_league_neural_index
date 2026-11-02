import numpy as np
import polars as pl
import torch
import torch.nn as nn
from dataclasses import dataclass
from props import (BALL_FEATURES, BIG_BOOST_COLUMNS, BLUE_GOAL, GOAL_COLUMNS,
                   ORANGE_GOAL, PLAYER_FEATURES, PLAYER_SLOTS, SMALL_BOOST_COLUMNS,
                   normalize_model_entities)

CAMERA_ENTITY_COUNT = 43
CAMERA_FEATURE_COUNT = 22
CAMERA_ENTITY_TYPES = np.asarray([0] + [1] * 6 + [2] * 2 + [3] * 34, dtype=np.int64)
CAMERA_RADII = np.asarray([93.15] + [70.0] * 6 + [120.0] * 2 + [70.0] * 6 + [35.0] * 28, dtype=np.float32)
CAMERA_FEATURE_SCALE = np.asarray(
    [5000.0, 5000.0, 5000.0, 2500.0, 2500.0, 2500.0, 1.0, 1.0, 5000.0, 1.0, 1.0,
     110.0, 300.0, 100.0, 30.0, 1.0, 10.0, 3.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)


def camera_entity_mask(model_entities='all'):
    selected = set(normalize_model_entities(model_entities))
    mask = np.zeros(CAMERA_ENTITY_COUNT, dtype=bool)
    if 'ball' in selected:
        mask[0] = True
    if 'players' in selected:
        mask[1:7] = True
    if 'goals' in selected:
        mask[7:9] = True
    if 'big_boosts' in selected:
        mask[9:15] = True
    if 'small_boosts' in selected:
        mask[15:43] = True
    return mask

def _numeric_columns(frame, names, default=0.0):
    values = []
    for name in names:
        if name not in frame.columns:
            values.append(np.full(frame.height, default, dtype=np.float32))
        else:
            values.append(frame[name].cast(pl.Float32, strict=False)
                          .fill_null(default).to_numpy().astype(np.float32, copy=False))
    return np.stack(values, axis=-1)


def _unit(values, fallback):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.where(norm > 1e-6, values / np.maximum(norm, 1e-6), fallback)


def _forward_from_quaternion(quaternion):
    # Source quaternions are stored as x, y, z, w; local +X is car forward.
    length = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    q = quaternion / np.maximum(length, 1e-6)
    x, y, z, w = (q[..., index] for index in range(4))
    return np.stack((1.0 - 2.0 * (y * y + z * z),
                     2.0 * (x * y + z * w),
                     2.0 * (x * z - y * w)), axis=-1)


def camera_pose(players, ball, *, mode, distance, height, angle_degrees, behind=None,
                frame_pitch=128.0, frame_yaw=128.0, using_swivel=False):
    position = players[..., :3].astype(np.float32)
    shape = position.shape[:2]
    if isinstance(mode, str):
        if mode not in ('car', 'ball'):
            raise ValueError('player_view_camera_mode must be car or ball')
        ball_mode = np.full(shape, mode == 'ball', dtype=bool)
    else:
        ball_mode = np.broadcast_to(np.asarray(mode, dtype=bool), shape)
    distance = np.broadcast_to(np.asarray(distance, dtype=np.float32), shape)
    height = np.broadcast_to(np.asarray(height, dtype=np.float32), shape)
    angle_degrees = np.broadcast_to(np.asarray(angle_degrees, dtype=np.float32), shape)
    behind = (np.zeros(shape, dtype=bool) if behind is None else
              np.broadcast_to(np.asarray(behind, dtype=bool), shape))
    frame_pitch = np.broadcast_to(np.asarray(frame_pitch, dtype=np.float32), shape)
    frame_yaw = np.broadcast_to(np.asarray(frame_yaw, dtype=np.float32), shape)
    using_swivel = np.broadcast_to(np.asarray(using_swivel, dtype=bool), shape)
    forward = _forward_from_quaternion(players[..., 9:13])
    horizontal = forward.copy()
    horizontal[..., 2] = 0.0
    fallback = np.zeros_like(horizontal)
    fallback[..., 0] = 1.0
    horizontal = _unit(horizontal, fallback)
    horizontal = np.where(behind[..., None], -horizontal, horizontal)
    up_world = np.zeros_like(horizontal)
    up_world[..., 2] = 1.0
    focus = position + up_world * height[..., None]
    yaw_offset = ((frame_yaw - 128.0) / 128.0) * np.pi
    pitch_offset = np.clip(((frame_pitch - 128.0) / 128.0) * (np.pi / 2.0),
                           -np.radians(80.0), np.radians(80.0))
    manual_swivel = using_swivel | (np.abs(yaw_offset) > 1e-4) | (np.abs(pitch_offset) > 1e-4)
    swivel_yaw = np.where(manual_swivel, yaw_offset, 0.0)
    swivel_horizontal = np.stack((
        horizontal[..., 0] * np.cos(swivel_yaw) - horizontal[..., 1] * np.sin(swivel_yaw),
        horizontal[..., 0] * np.sin(swivel_yaw) + horizontal[..., 1] * np.cos(swivel_yaw),
        np.zeros(shape, dtype=np.float32),
    ), axis=-1)
    camera_angle = np.radians(angle_degrees) + np.where(manual_swivel, pitch_offset, 0.0)
    car_direction = _unit(swivel_horizontal * np.cos(camera_angle)[..., None]
                           + up_world * np.sin(camera_angle)[..., None], horizontal)
    to_ball = ball[:, None, :3] - focus
    ball_horizontal = to_ball.copy()
    ball_horizontal[..., 2] = 0.0
    ball_distance = np.linalg.norm(ball_horizontal, axis=-1)
    ball_horizontal = _unit(ball_horizontal, horizontal)
    ball_pitch = np.arctan2(to_ball[..., 2], ball_distance)
    pitch_blend = np.clip((np.abs(ball_pitch) - np.radians(22.0)) / np.radians(22.0), 0.0, 1.0)
    target_pitch = camera_angle + (ball_pitch - camera_angle) * (pitch_blend * 0.8)
    ball_direction = _unit(ball_horizontal * np.cos(target_pitch)[..., None]
                           + up_world * np.sin(target_pitch)[..., None], car_direction)
    direction = np.where((ball_mode & ~manual_swivel)[..., None], ball_direction, car_direction)
    origin = focus - direction * distance[..., None]
    right = _unit(np.cross(direction, up_world), np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
    up = _unit(np.cross(right, direction), up_world)
    return origin, direction, right, up


def _arena_clear(origin, target):
    delta = target - origin
    clear = np.ones(delta.shape[:-1], dtype=bool)
    for axis, boundary in ((0, -4096.0), (0, 4096.0),
                           (1, -5120.0), (1, 5120.0),
                           (2, 0.0), (2, 2048.0)):
        component = delta[..., axis]
        fraction = (boundary - origin[..., axis]) / np.where(np.abs(component) > 1e-6, component, 1.0)
        crossed = (np.abs(component) > 1e-6) & (fraction > 1e-5) & (fraction < 1.0 - 1e-5)
        intersection = origin + fraction[..., None] * delta
        if axis == 0:
            crossed &= (np.abs(intersection[..., 1]) <= 5120.0) & (intersection[..., 2] >= 0.0)
        elif axis == 1:
            crossed &= (np.abs(intersection[..., 0]) <= 4096.0) & (intersection[..., 2] >= 0.0)
            crossed &= ~((np.abs(intersection[..., 0]) < 892.755)
                         & (intersection[..., 2] < 642.775))
        else:
            crossed &= (np.abs(intersection[..., 0]) <= 4096.0) & (np.abs(intersection[..., 1]) <= 5120.0)
        clear &= ~crossed
    return clear


def _object_clear(origin, target, positions, present):
    delta = target - origin
    distance_squared = np.maximum(np.sum(delta * delta, axis=-1), 1e-6)
    clear = np.ones(distance_squared.shape, dtype=bool)
    for occluder in range(7):
        center = positions[:, None, None, occluder, :]
        along = np.sum((center - origin) * delta, axis=-1) / distance_squared
        closest = origin + along[..., None] * delta
        radial = np.sum((closest - center) ** 2, axis=-1)
        blocked = present[:, occluder, None, None] & (along > 1e-4) & (along < 0.98)
        blocked &= radial < CAMERA_RADII[occluder] ** 2
        clear &= ~blocked
    return clear


def camera_frame_observations(frame, *, mode='recorded', fov_degrees=110.0, distance=270.0,
                              height=100.0, angle_degrees=-3.0, aspect_ratio=16.0 / 9.0,
                              model_entities='all'):
    if not (0.0 < fov_degrees < 180.0 and aspect_ratio > 0.0 and distance > 0.0):
        raise ValueError('invalid camera FOV, aspect ratio, or distance')
    if mode not in ('recorded', 'car', 'ball'):
        raise ValueError('player_view_camera_mode must be recorded, car, or ball')
    required = ['replay_id', 'stint_id', '_source_time', *BALL_FEATURES[:6],
                *(f'player_{slot}_{feature}' for slot in PLAYER_SLOTS
                  for feature in (*PLAYER_FEATURES[:6], *PLAYER_FEATURES[9:13], 'on_field', 'demoed'))]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f'camera view requires physical state columns: {sorted(missing)[:8]}')
    camera_defaults = {
        'camera_fov': fov_degrees, 'camera_distance': distance,
        'camera_height': height, 'camera_angle': angle_degrees,
        'camera_stiffness': 0.5, 'camera_swivel': 5.0,
        'camera_transition': 1.2, 'camera_using_secondary_camera': 0.0,
        'camera_using_behind_view': 0.0, 'camera_using_swivel': 0.0,
        'camera_frame_pitch': 128.0, 'camera_frame_yaw': 128.0,
    }
    if mode == 'recorded':
        missing_camera = [f'player_{slot}_{field}' for slot in PLAYER_SLOTS
                          for field in ('camera_fov', 'camera_distance', 'camera_height',
                                        'camera_angle', 'camera_using_secondary_camera')
                          if f'player_{slot}_{field}' not in frame.columns]
        if missing_camera:
            raise ValueError(f'recorded camera mode requires replay camera fields: {missing_camera[:8]}')
    camera = {
        field: np.stack([
            _numeric_columns(frame, (f'player_{slot}_{field}',), default=value)[:, 0]
            for slot in PLAYER_SLOTS
        ], axis=1)
        for field, value in camera_defaults.items()
    }
    for field in ('camera_fov', 'camera_distance', 'camera_height', 'camera_angle'):
        value = camera[field]
        camera[field] = np.where(np.isfinite(value), value, camera_defaults[field])
    camera['camera_fov'] = np.clip(np.where((camera['camera_fov'] > 0) &
                                             (camera['camera_fov'] < 180),
                                             camera['camera_fov'], fov_degrees), 60.0, 120.0)
    camera['camera_distance'] = np.maximum(camera['camera_distance'], 0.0)
    ball = _numeric_columns(frame, BALL_FEATURES[:6])
    players = np.stack([
        _numeric_columns(frame, tuple(f'player_{slot}_{feature}' for feature in PLAYER_FEATURES))
        for slot in PLAYER_SLOTS
    ], axis=1)
    goals = (_numeric_columns(frame, GOAL_COLUMNS).reshape(frame.height, 2, 9)
             if set(GOAL_COLUMNS).issubset(frame.columns)
             else np.broadcast_to(np.stack((BLUE_GOAL, ORANGE_GOAL)), (frame.height, 2, 9)).copy())
    if not set(GOAL_COLUMNS).issubset(frame.columns) and '_rotate_180' in frame.columns:
        signs = np.where(frame['_rotate_180'].fill_null(False).to_numpy()[:, None, None],
                         (-1.0, -1.0, 1.0), (1.0, 1.0, 1.0)).astype(np.float32)
        goals[..., :3] *= signs
        goals[..., 6:9] *= signs
    if not (set(BIG_BOOST_COLUMNS) | set(SMALL_BOOST_COLUMNS)).issubset(frame.columns):
        raise ValueError('camera view requires all 34 boost-pad states')
    boosts = np.concatenate((
        _numeric_columns(frame, BIG_BOOST_COLUMNS).reshape(frame.height, 6, 6),
        _numeric_columns(frame, SMALL_BOOST_COLUMNS).reshape(frame.height, 28, 6),
    ), axis=1)
    positions = np.concatenate((
        ball[:, None, :3], players[..., :3], goals[..., :3],
        np.concatenate((boosts[..., :2], np.concatenate((
            np.full((frame.height, 6, 1), 73.0, dtype=np.float32),
            np.full((frame.height, 28, 1), 32.0, dtype=np.float32),
        ), axis=1)), axis=-1),
    ), axis=1)
    velocities = np.concatenate((
        ball[:, None, 3:6], players[..., 3:6],
        np.zeros((frame.height, 36, 3), dtype=np.float32),
    ), axis=1)
    player_present = (players[..., PLAYER_FEATURES.index('on_field')] > 0.5
                      ) & ~(players[..., PLAYER_FEATURES.index('demoed')] > 0.5)
    present = np.concatenate((
        np.ones((frame.height, 1), dtype=bool), player_present,
        np.ones((frame.height, 36), dtype=bool),
    ), axis=1)
    ball_mode = (camera['camera_using_secondary_camera'] > 0.5 if mode == 'recorded'
                 else np.full((frame.height, 6), mode == 'ball', dtype=bool))
    origin, direction, right, up = camera_pose(
        players, ball, mode=ball_mode, distance=camera['camera_distance'],
        height=camera['camera_height'], angle_degrees=camera['camera_angle'],
        behind=camera['camera_using_behind_view'] > 0.5,
        frame_pitch=camera['camera_frame_pitch'], frame_yaw=camera['camera_frame_yaw'],
        using_swivel=camera['camera_using_swivel'] > 0.5,
    )
    origin_expanded = origin[:, :, None, :]
    relative = positions[:, None, :, :] - origin_expanded
    depth = np.sum(relative * direction[:, :, None, :], axis=-1)
    camera_x = np.sum(relative * right[:, :, None, :], axis=-1)
    camera_y = np.sum(relative * up[:, :, None, :], axis=-1)
    vertical_tangent = np.tan(np.radians(camera['camera_fov']) / 2.0)[:, :, None]
    horizontal_tangent = vertical_tangent * aspect_ratio
    safe_depth = np.maximum(depth, 1e-3)
    screen_x = camera_x / (safe_depth * horizontal_tangent)
    screen_y = camera_y / (safe_depth * vertical_tangent)
    size_x = CAMERA_RADII[None, None, :] / (safe_depth * horizontal_tangent)
    size_y = CAMERA_RADII[None, None, :] / (safe_depth * vertical_tangent)
    size = size_y
    frustum = (depth > 1.0) & (np.abs(screen_x) <= 1.0 + size_x)
    frustum &= np.abs(screen_y) <= 1.0 + size_y
    clear = _arena_clear(origin_expanded, positions[:, None, :, :])
    clear &= _object_clear(origin_expanded, positions[:, None, :, :], positions, present)
    finite = np.isfinite(positions).all(axis=-1)
    finite &= np.isfinite(velocities).all(axis=-1)
    viewer_valid = player_present & np.isfinite(origin).all(axis=-1)
    visible = viewer_valid[:, :, None] & present[:, None, :] & finite[:, None, :]
    visible &= frustum & clear
    visible &= camera_entity_mask(model_entities)[None, None, :]
    camera_velocity = players[..., 3:6]
    replay = frame['replay_id'].to_list()
    stint = frame['stint_id'].to_list()
    times = frame['_source_time'].to_numpy().astype(np.float64)
    identities = [
        (frame[f'player_{slot}_id'].fill_null('').cast(pl.String).to_list()
         if f'player_{slot}_id' in frame.columns else [f'slot:{slot}'] * frame.height)
        for slot in PLAYER_SLOTS
    ]
    identities = np.asarray(identities, dtype=object).T
    camera_velocity = camera_velocity.copy()
    for row in range(1, frame.height):
        if replay[row] != replay[row - 1] or stint[row] != stint[row - 1]:
            continue
        delta_time = times[row] - times[row - 1]
        if delta_time <= 0:
            continue
        same_identity = identities[row] == identities[row - 1]
        camera_velocity[row, same_identity] = (
            origin[row, same_identity] - origin[row - 1, same_identity]
        ) / delta_time
    relative_velocity = velocities[:, None, :, :] - camera_velocity[:, :, None, :]
    velocity_x = np.sum(relative_velocity * right[:, :, None, :], axis=-1)
    velocity_y = np.sum(relative_velocity * up[:, :, None, :], axis=-1)
    velocity_z = np.sum(relative_velocity * direction[:, :, None, :], axis=-1)
    status = np.ones((frame.height, CAMERA_ENTITY_COUNT), dtype=np.float32)
    status[:, 9:] = boosts[..., 4]
    metadata = np.stack((
        camera['camera_fov'], camera['camera_distance'], camera['camera_height'],
        camera['camera_angle'], camera['camera_stiffness'], camera['camera_swivel'],
        camera['camera_transition'], ball_mode.astype(np.float32),
        camera['camera_using_behind_view'],
        (camera['camera_frame_pitch'] - 128.0) / 127.0,
        (camera['camera_frame_yaw'] - 128.0) / 127.0,
    ), axis=-1)
    metadata = np.broadcast_to(metadata[:, :, None, :], (frame.height, 6, CAMERA_ENTITY_COUNT, 11))
    features = np.concatenate((np.stack((
        camera_x, camera_y, depth, velocity_x, velocity_y, velocity_z,
        screen_x, screen_y, np.linalg.norm(relative, axis=-1), size,
        np.broadcast_to(status[:, None, :], size.shape),
    ), axis=-1), metadata), axis=-1).astype(np.float32)
    features = np.nan_to_num(features / CAMERA_FEATURE_SCALE, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.where(visible[..., None], features, 0.0)
    team_ids = np.stack([
        _numeric_columns(frame, (f'player_{slot}_team_id',), default=2.0)[:, 0]
        for slot in PLAYER_SLOTS
    ], axis=1)
    entity_team = np.concatenate((
        np.full((frame.height, 1), 2.0, dtype=np.float32), team_ids,
        np.broadcast_to(np.asarray([[0.0, 1.0]], dtype=np.float32), (frame.height, 2)),
        np.full((frame.height, 34), 2.0, dtype=np.float32),
    ), axis=1)
    relation = np.where(entity_team[:, None, :] == team_ids[:, :, None], 1, 2)
    relation = np.where(entity_team[:, None, :] == 2, 3, relation)
    for player in range(6):
        relation[:, player, player + 1] = 0
    return features, visible, viewer_valid, relation.astype(np.int64), times, replay, stint, identities


def camera_histories(frame, indices, *, history_seconds=2.0, sample_rate=10,
                     mode='recorded', fov_degrees=110.0, distance=270.0,
                     height=100.0, angle_degrees=-3.0, aspect_ratio=16.0 / 9.0,
                     observations=None, model_entities='all'):
    if history_seconds <= 0 or sample_rate <= 0:
        raise ValueError('camera history_seconds and sample_rate must be positive')
    indices = np.asarray(tuple(indices), dtype=np.int64)
    if observations is None:
        observations = camera_frame_observations(
            frame, mode=mode, fov_degrees=fov_degrees, distance=distance, height=height,
            angle_degrees=angle_degrees, aspect_ratio=aspect_ratio,
            model_entities=model_entities,
        )
    features, visible, viewer_valid, relation, times, replay, stint, identities = observations
    selected_entities = camera_entity_mask(model_entities)
    entity_indices = np.flatnonzero(selected_entities)
    steps = int(round(history_seconds * sample_rate)) + 1
    rows = np.full((indices.size, steps), -1, dtype=np.int64)
    ages = np.zeros((indices.size, steps), dtype=np.float32)
    groups = {}
    for row, key in enumerate(zip(replay, stint)):
        groups.setdefault(key, []).append(row)
    groups = {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}
    offsets = np.arange(steps - 1, -1, -1, dtype=np.float64) / sample_rate
    for output, current in enumerate(indices):
        group = groups[(replay[current], stint[current])]
        group = group[group <= current]
        targets = times[current] - offsets
        candidates = np.searchsorted(times[group], targets, side='right') - 1
        valid = candidates >= 0
        chosen = group[np.maximum(candidates, 0)]
        valid &= times[current] - times[chosen] <= history_seconds + 1e-5
        valid &= targets - times[chosen] <= 1.0 / sample_rate + 1e-5
        rows[output, valid] = chosen[valid]
        ages[output, valid] = (times[current] - times[chosen[valid]]).astype(np.float32)
    selected = np.maximum(rows, 0)
    observed = visible[selected][..., entity_indices]
    valid = viewer_valid[selected] & (rows[:, :, None] >= 0)
    valid = valid.transpose(0, 2, 1)
    same_viewer = (identities[selected] == identities[indices, None, :]).transpose(0, 2, 1)
    valid &= same_viewer & viewer_valid[indices, :, None]
    entity_valid = np.broadcast_to(valid[..., None], (indices.size, 6, steps, entity_indices.size)).copy()
    player_present = viewer_valid[:, :6]
    for entity in range(6):
        selected_position = np.flatnonzero(entity_indices == entity + 1)
        if not selected_position.size:
            continue
        selected_position = selected_position[0]
        same_entity = identities[selected][..., entity] == identities[indices, None, entity]
        entity_valid[:, :, :, selected_position] &= same_entity[:, None, :]
        entity_valid[:, :, :, selected_position] &= player_present[selected][..., entity][:, None, :]
    observed = observed.transpose(0, 2, 1, 3) & entity_valid
    history_features = features[selected][..., entity_indices, :].transpose(0, 2, 1, 3, 4)
    history_features = np.where(observed[..., None], history_features, 0.0)
    return {
        'camera_features': torch.from_numpy(history_features.astype(np.float32, copy=False)),
        'camera_visible': torch.from_numpy(observed),
        'camera_valid': torch.from_numpy(entity_valid),
        'camera_age': torch.from_numpy(ages),
        'camera_relation': torch.from_numpy(relation[indices][:, :, entity_indices]),
    }


class PlayerCameraViewEncoder(nn.Module):
    @dataclass(frozen=True)
    class Settings:
        history_seconds: float = 2.0
        sample_rate: int = 10
        temporal_blocks: int = 2
        attention_heads: int = 8
        feedforward_dim: int = 256
        dropout: float = 0.05
        camera_mode: str = 'recorded'
        fov_degrees: float = 110.0
        distance: float = 270.0
        height: float = 100.0
        angle_degrees: float = -3.0
        aspect_ratio: float = 16.0 / 9.0

        @classmethod
        def from_saved(cls, value):
            if isinstance(value, cls):
                values = dict(vars(value))
            elif isinstance(value, dict):
                values = dict(value)
            else:
                values = dict(vars(value))
            if 'sample_rate' not in values and 'hz' in values:
                values['sample_rate'] = values['hz']
            allowed = cls.__dataclass_fields__
            return cls(**{key: val for key, val in values.items() if key in allowed})

        def __post_init__(self):
            if self.history_seconds <= 0 or self.sample_rate <= 0 or self.temporal_blocks < 1:
                raise ValueError('camera history and temporal blocks must be positive')
            if self.attention_heads < 1 or self.feedforward_dim < 1:
                raise ValueError('camera attention dimensions must be positive')
            if not 0.0 <= self.dropout < 1.0:
                raise ValueError('camera dropout must lie in [0, 1)')
            if self.camera_mode not in ('recorded', 'car', 'ball'):
                raise ValueError('camera_mode must be recorded, car, or ball')
            if not (0 < self.fov_degrees < 180 and self.distance > 0 and self.aspect_ratio > 0):
                raise ValueError('invalid camera geometry')

    def __init__(self, d_model=128, settings=None, entity_count=CAMERA_ENTITY_COUNT,
                 model_entities='all'):
        super().__init__()
        self.settings = settings or self.Settings()
        temporal_blocks = self.settings.temporal_blocks
        attention_heads = self.settings.attention_heads
        dropout = self.settings.dropout
        feedforward_dim = self.settings.feedforward_dim
        if d_model % attention_heads:
            raise ValueError('camera attention_heads must divide d_model')
        self.d_model = d_model
        entity_indices = np.flatnonzero(camera_entity_mask(model_entities))
        if entity_count != CAMERA_ENTITY_COUNT:
            entity_indices = entity_indices[:entity_count]
        self.entity_count = len(entity_indices)
        self.register_buffer('entity_indices', torch.from_numpy(entity_indices), persistent=False)
        self.register_buffer(
            'entity_types', torch.as_tensor(CAMERA_ENTITY_TYPES[entity_indices]), persistent=False
        )
        self.entity_encoder = nn.Sequential(
            nn.Linear(CAMERA_FEATURE_COUNT, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.type_embedding = nn.Embedding(4, d_model)
        self.relation_embedding = nn.Embedding(4, d_model)
        # Keep the full identity table for compatibility with existing camera
        # checkpoints, while inputs contain only selected entities.
        self.identity_embedding = nn.Embedding(CAMERA_ENTITY_COUNT, d_model)
        self.visibility_embedding = nn.Embedding(2, d_model)
        self.time_projection = nn.Linear(1, d_model)
        self.temporal_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, attention_heads, feedforward_dim, dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(temporal_blocks)
        ])
        self.query = nn.Linear(d_model, d_model)
        self.pool = nn.MultiheadAttention(d_model, attention_heads, dropout=dropout, batch_first=True)
        self.modulation = nn.Linear(d_model, 2 * d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, player_embedding, features, visible, valid, age, relation):
        batch, players, steps, entities, feature_count = features.shape
        if (players != 6 or entities != self.entity_count or feature_count != CAMERA_FEATURE_COUNT
                or visible.shape != (batch, players, steps, entities)
                or valid.shape != visible.shape or age.shape != (batch, steps)):
            raise ValueError('camera inputs do not match the configured history dimensions')
        visible = visible.bool() & valid.bool()
        valid = valid.bool()
        selected_entities = torch.ones((1, 1, 1, self.entity_count),
                                       device=features.device, dtype=torch.bool)
        visible &= selected_entities
        valid &= selected_entities
        observed = self.entity_encoder(features * visible[..., None])
        observed = torch.where(visible[..., None], observed, torch.zeros_like(observed))
        types = self.entity_types
        static = (self.type_embedding(types)[None, None, None]
                  + self.identity_embedding(self.entity_indices)[None, None, None]
                  + self.relation_embedding(relation.long())[:, :, None])
        time = self.time_projection(age[:, None, :, None, None].to(features.dtype))
        sequence = observed + static + self.visibility_embedding(visible.long()) + time
        sequence = sequence * valid[..., None]
        sequence = sequence.permute(0, 1, 3, 2, 4).reshape(-1, steps, self.d_model)
        step_valid = valid.permute(0, 1, 3, 2).reshape(-1, steps)
        ever_visible = visible.any(dim=2).reshape(-1)
        encoded = features.new_zeros((batch * players * entities, self.d_model))
        if ever_visible.any():
            selected = sequence[ever_visible]
            selected_valid = step_valid[ever_visible]
            attention_valid = selected_valid.clone()
            attention_valid[:, 0] = True
            causal = torch.ones((steps, steps), dtype=torch.bool, device=features.device).triu(1)
            for block in self.temporal_blocks:
                selected = block(selected, src_mask=causal, src_key_padding_mask=~attention_valid)
            last = torch.arange(steps, device=features.device).view(1, -1)
            last = torch.where(selected_valid, last, 0).max(dim=1).values
            encoded[ever_visible] = selected[torch.arange(selected.shape[0], device=features.device), last]
        encoded = encoded.reshape(batch * players, entities, self.d_model)
        pool_valid = ever_visible.reshape(batch * players, entities)
        camera = encoded.new_zeros((batch * players, self.d_model))
        nonempty = pool_valid.any(dim=1)
        if nonempty.any():
            query = self.query(player_embedding.reshape(batch * players, self.d_model)[nonempty])[:, None]
            pooled, _ = self.pool(query, encoded[nonempty], encoded[nonempty],
                                  key_padding_mask=~pool_valid[nonempty], need_weights=False)
            camera[nonempty] = pooled[:, 0].to(camera.dtype)
        camera = camera.reshape(batch, players, self.d_model)
        gamma, beta = self.modulation(camera).chunk(2, dim=-1)
        gate = torch.sigmoid(self.gate(torch.cat((player_embedding, camera), dim=-1)))
        fused = player_embedding + gate * (gamma * player_embedding + beta)
        return torch.where(nonempty.reshape(batch, players, 1), fused, player_embedding)
