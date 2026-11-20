import hashlib
import numpy as np
from pathlib import Path
from typing import Literal, get_args

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "lgps" / "train.parquet"

OptimizerName = Literal["AdamW", "Adam", "SGD", "RMSprop", "Adagrad", "Hybrid"]


def build_optimizers(model, config, device, *, adamw_modules=()):
    import torch

    name = config.optimizer
    choices = get_args(OptimizerName)

    if not isinstance(name, str) or name not in choices:
        raise ValueError(f'optimizer must be one of: {", ".join(choices)}')

    optimizers = []

    if name == 'Hybrid':
        muon_type = getattr(torch.optim, 'Muon', None)

        if muon_type is None:
            raise RuntimeError('Hybrid optimizer requires torch.optim.Muon')

        adamw_ids = {id(parameter) for module in adamw_modules for parameter in module.parameters()}
        muon_parameters = []
        adamw_parameters = []

        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            if (parameter.ndim == 2 and parameter_name.endswith('.weight')
                    and 'embedding' not in parameter_name and '.head.' not in parameter_name
                    and id(parameter) not in adamw_ids):
                muon_parameters.append(parameter)
            else:
                adamw_parameters.append(parameter)

        if not muon_parameters or not adamw_parameters:
            raise RuntimeError('Hybrid optimizer requires matrix weights and other trainable parameters')

        optimizers.append(muon_type(muon_parameters, lr=config.muon_learning_rate, weight_decay=config.weight_decay))
        name = 'AdamW'
        parameters = adamw_parameters
    else:
        parameters = model.parameters()

    optimizer_type = getattr(torch.optim, name)
    kwargs = {'lr': config.learning_rate, 'weight_decay': config.weight_decay}

    if device.type == 'cuda' and name == 'AdamW':
        # Materialize the parameters so a failed fused construction can be retried.
        parameters = list(parameters)

        try:
            optimizer = optimizer_type(parameters, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            optimizer = optimizer_type(parameters, **kwargs)
    else:
        optimizer = optimizer_type(parameters, **kwargs)

    optimizers.append(optimizer)
    return optimizers

PLAYER_FEATURES = (
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "rot_x", "rot_y", "rot_z", "rot_w",
    "boost_raw", "respawn_timer", "demoed", "on_field",
    "throttle", "steer", "handbrake", "pitch", "yaw", "roll",
    "is_boosting", "is_holding_jump", "supersonic", "has_dodge", "is_single_jumping",
    "is_double_jumping", "is_dodging",
    "touching_ball",
    "in_air", "on_ground", "on_ceiling", "on_backwall", "on_sidewall",
    "dodge_torque_x", "dodge_torque_y", "dodge_torque_z",
    "dodge_impulse_x", "dodge_impulse_y", "dodge_impulse_z",
    "double_jump_impulse_x", "double_jump_impulse_y", "double_jump_impulse_z",
    "is_driving", "is_powersliding", "is_throttling", "in_reverse",
    "is_picking_up_big_boost", "is_picking_up_small_boost",
)

PLAYER_CONTROL_COLUMNS = {
    'is_holding_jump': ('is_holding_jump', 'control_jump', 'input_jump'),
    'is_boosting': ('is_boosting', 'control_boost', 'input_boost', 'boost_input'),
    'handbrake': ('handbrake', 'control_handbrake', 'input_handbrake', 'is_powersliding'),
    'throttle': ('throttle', 'throttle_scaled', 'control_throttle', 'input_throttle'),
    'steer': ('steer', 'steer_scaled', 'control_steer', 'input_steer'),
    **{axis: (f'control_{axis}', f'input_{axis}', f'{axis}_input', f'controls_{axis}', axis)
       for axis in ('pitch', 'yaw', 'roll')},
}

BALL_FEATURES = (
    "ball_pos_x", "ball_pos_y", "ball_pos_z",
    "ball_vel_x", "ball_vel_y", "ball_vel_z",
    "ball_ang_vel_x", "ball_ang_vel_y", "ball_ang_vel_z",
    "ball_rot_x", "ball_rot_y", "ball_rot_z", "ball_rot_w",
)

GOAL_FEATURES = (
    "pos_x", "pos_y", "pos_z",
    "dimensions_x", "dimensions_y", "dimensions_z",
    "opening_direction_x", "opening_direction_y", "opening_direction_z",
)

CONTEXT_FEATURES = (
    "game_seconds_remaining",
    "kickoff_in_progress",
    "is_overtime",
    "blue_score",
    "orange_score",
)

PLAYER_SLOTS = tuple(range(1, 7))
GOAL_COLUMNS = tuple(
    f"{team}_goal_{feature}"
    for team in ("blue", "orange")
    for feature in GOAL_FEATURES
)
BLUE_GOAL = np.asarray((0., -5120., 321., 1785.5, 880., 642.775, 0., 1., 0.), dtype=np.float32)
ORANGE_GOAL = np.asarray((0., 5120., 321., 1785.5, 880., 642.775, 0., -1., 0.), dtype=np.float32)
BOOST_FEATURES = ("pos_x", "pos_y", "pos_z", "big_boost", "small_boost", "available", "refill_timer")
BOOST_SCALE_WITHOUT_REFILL = (5000.0, 5000.0, 5000.0, 1.0, 1.0, 1.0)
BOOST_COLUMNS = tuple(f"boost_pad_{index:02d}_{feature}" for index in range(34) for feature in BOOST_FEATURES)
BIG_BOOST_COLUMNS = BOOST_COLUMNS[:6 * len(BOOST_FEATURES)]
SMALL_BOOST_COLUMNS = BOOST_COLUMNS[6 * len(BOOST_FEATURES):]
BOOST_ENTITY_GROUPS = (
    ("big_boosts", 6, BIG_BOOST_COLUMNS, 10.0),
    ("small_boosts", 28, SMALL_BOOST_COLUMNS, 4.0),
)

MAX_PLAYERS_PER_TEAM = 3

MODEL_ENTITY_NAMES = ("players", "ball", "goals", "big_boosts", "small_boosts", "context")

def replay_split(replay_id, train_split, val_split, source_name=''):
    if '2026' in str(source_name).replace('\\', '/').split('/'):
        return 'test'

    digest = hashlib.blake2b(str(replay_id).encode('utf-8'), digest_size=8).digest()
    value = int.from_bytes(digest, 'big') / float(1 << 64)

    if value < train_split:
        return 'train'

    if value < train_split + val_split:
        return 'validation'

    return 'test'


class BatchDevice:
    @staticmethod
    def move(batch, device):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def quaternion_to_rocket_league_euler(quaternion_xyzw):
    """Convert game xyzw quaternions to Rocket League pitch/yaw/roll radians."""
    quaternion = np.asarray(quaternion_xyzw)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    sin_pitch = 2.0 * (w * y - z * x)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    pitch = -np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(sin_yaw, cos_yaw)
    roll = -np.arctan2(sin_roll, cos_roll)
    return np.stack((pitch, yaw, roll), axis=-1)

def normalize_model_entities(value):
    if value == "all":
        return MODEL_ENTITY_NAMES

    if isinstance(value, list):
        value = tuple(value)

    if not isinstance(value, tuple) or not value:
        raise ValueError("model_entities must be 'all' or a non-empty list")

    unknown = set(value).difference(MODEL_ENTITY_NAMES)

    if unknown or len(set(value)) != len(value):
        raise ValueError(f"invalid or duplicate model_entities: {sorted(unknown)}")

    return tuple(name for name in MODEL_ENTITY_NAMES if name in value)
