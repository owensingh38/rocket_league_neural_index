import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "lgps" / "train.parquet"

PLAYER_FEATURES = (
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "rot_x", "rot_y", "rot_z", "rot_w",
    "boost_scaled", "respawn_timer", "demoed", "on_field",
    "throttle", "steer", "handbrake", "pitch", "yaw", "roll",
    "is_boosting", "supersonic", "has_dodge", "is_jumping",
    "is_double_jumping", "is_dodging",
)

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
    "seconds_elapsed",
    "overtime",
    "blue_team_score",
    "orange_team_score",
)

PLAYER_SLOTS = tuple(range(1, 7))
GOAL_COLUMNS = tuple(
    f"{team}_goal_{feature}"
    for team in ("blue", "orange")
    for feature in GOAL_FEATURES
)
BLUE_GOAL = np.asarray((0., -5120., 321., 1785.5, 880., 642.775, 0., 1., 0.), dtype=np.float32)
ORANGE_GOAL = np.asarray((0., 5120., 321., 1785.5, 880., 642.775, 0., -1., 0.), dtype=np.float32)
BOOST_FEATURES = ("pos_x", "pos_y", "big_boost", "small_boost", "available", "refill_timer")
BOOST_COLUMNS = tuple(f"boost_pad_{index:02d}_{feature}" for index in range(34) for feature in BOOST_FEATURES)
BIG_BOOST_COLUMNS = BOOST_COLUMNS[:6 * len(BOOST_FEATURES)]
SMALL_BOOST_COLUMNS = BOOST_COLUMNS[6 * len(BOOST_FEATURES):]
BOOST_ENTITY_GROUPS = (
    ("big_boosts", 6, BIG_BOOST_COLUMNS, 10.0),
    ("small_boosts", 28, SMALL_BOOST_COLUMNS, 5.0),
)

MAX_PLAYERS_PER_TEAM = 3

MODEL_ENTITY_NAMES = ("players", "ball", "goals", "big_boosts", "small_boosts", "context")

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
