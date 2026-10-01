import math
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

GEOMETRY_DIM = 156
DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "lgps" / "train.parquet"
OUTPUT_DIR = DATA_DIR / "lgps" / "output"

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
GOAL_STATE_COLUMNS = (
    "blue_goal_in_next_ten_seconds",
    "orange_goal_in_next_ten_seconds",
    "neither_goal_in_next_ten_seconds",
)
GOAL_COLUMNS = tuple(
    f"{team}_goal_{feature}"
    for team in ("blue", "orange")
    for feature in GOAL_FEATURES
)
BLUE_GOAL = np.asarray((0., -5120., 321., 1785.5, 880., 642.775, 0., 1., 0.), dtype=np.float32)
ORANGE_GOAL = np.asarray((0., 5120., 321., 1785.5, 880., 642.775, 0., -1., 0.), dtype=np.float32)
BOOST_FEATURES = ("pos_x", "pos_y", "big_boost", "small_boost", "available", "refill_timer")
BOOST_POSITIONS = np.asarray([
    (-3072,-4096),(3072,-4096),(-3584,0),(3584,0),(-3072,4096),(3072,4096),
    (-1792,-4184),(0,-4240),(1792,-4184),(-940,-3308),(940,-3308),(0,-2816),
    (-3584,-2484),(-1788,-2300),(1788,-2300),(3584,-2484),(-2048,-1036),(0,-1024),(2048,-1036),
    (-1024,0),(1024,0),(-2048,1036),(0,1024),(2048,1036),(-3584,2484),(-1788,2300),(1788,2300),
    (3584,2484),(0,2816),(-940,3308),(940,3308),(-1792,4184),(0,4240),(1792,4184),
], dtype=np.float32)
BOOST_COLUMNS = tuple(f"boost_pad_{index:02d}_{feature}" for index in range(34) for feature in BOOST_FEATURES)
BIG_BOOST_COLUMNS = BOOST_COLUMNS[:6 * len(BOOST_FEATURES)]
SMALL_BOOST_COLUMNS = BOOST_COLUMNS[6 * len(BOOST_FEATURES):]
BOOST_ENTITY_GROUPS = (("big_boosts", 6, BIG_BOOST_COLUMNS, 10.0), ("small_boosts", 28, SMALL_BOOST_COLUMNS, 5.0))
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

@dataclass(frozen=True)
class LGPSModelConfig:
    d_model: int = 128
    num_heads: int = 8
    num_blocks: int = 4
    loops_per_block: int = 4
    ff_dim: int = 256
    expanded_geometry: bool = False
    pairwise_rank: int = 32
    dropout: float = 0.05
    targets_per_team: int = 10
    geometric_width: float = 1.25
    model_entities: str | tuple[str, ...] = "all"

    def __post_init__(self):
        object.__setattr__(self, "model_entities", normalize_model_entities(self.model_entities))
        if self.d_model <= 0 or self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if min(self.num_blocks, self.loops_per_block, self.ff_dim, self.pairwise_rank) <= 0:
            raise ValueError("spatial block counts, loop counts, ff_dim and pairwise_rank must be positive")
        if self.targets_per_team < 1:
            raise ValueError("targets_per_team must be at least one")
        if not math.isfinite(self.geometric_width) or self.geometric_width <= 0:
            raise ValueError("geometric_width must be finite and positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def geometry_dim(self) -> int:
        # Derived geometry width; controlled exclusively by expanded_geometry.
        return GEOMETRY_DIM if self.expanded_geometry else 15

    @property
    def num_outputs(self) -> int:
        return 2 * self.targets_per_team + 1


@dataclass(frozen=True)
class LGPSTrainConfig:
    input_path: Path = INPUT_DIR
    batch_size: int = 2048
    validation_batch_size: int = 8192
    effective_batch_size: int = 16384
    epochs: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    gradient_clip_norm: float = 1.0
    train_split: float = 0.9
    val_split: float = 0.0
    log_interval: int = 100
    def __post_init__(self):
        if not 0.0 <= self.train_split <= 1.0:
            raise ValueError("train_split must be between 0 and 1")
        if not 0.0 <= self.val_split <= 1.0:
            raise ValueError("val_split must be between 0 and 1")
        if self.train_split + self.val_split > 1.0:
            raise ValueError("train_split + val_split must not exceed 1")
        if self.batch_size <= 0 or self.validation_batch_size <= 0 or self.effective_batch_size <= 0:
            raise ValueError("batch sizes must be positive")

@dataclass(frozen=True)
class LGPSRuntimeConfig:
    device: str = "auto"
    world_size: int = 1
    ddp_backend: str = "nccl"
    find_unused_parameters: bool = False
    amp_policy: str = "auto"
    per_gpu_batch_size: int | None = None
    data_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    compile_mode: str | None = None
    activation_checkpointing: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "cuda"} or self.world_size not in {1, 2}:
            raise ValueError("device must be auto/cpu/cuda and world_size must be 1 or 2")
        if self.world_size == 2 and self.device == "cpu":
            raise ValueError("two-process LGPS requires CUDA/NCCL")
        if self.data_workers < 0 or self.prefetch_factor < 1:
            raise ValueError("data_workers must be nonnegative and prefetch_factor positive")
        if self.amp_policy != "auto" and not isinstance(getattr(torch, self.amp_policy, None), torch.dtype):
            raise ValueError("amp_policy must be auto or the name of a torch dtype")
        if self.wandb_project is None and any((self.wandb_entity, self.wandb_run_name, self.wandb_mode)):
            raise ValueError("W&B options require wandb_project")

def build_frame_batch(frame, indices, tokenizer, *, targets_per_team=10, geometric_width=1.25):
    indices = np.asarray(indices, dtype=np.int64)
    result = tokenizer(frame[indices.tolist()])
    result["temporal_targets"], result["collapsed_targets"] = temporal_targets(
        frame["time_until_next_goal"].to_numpy()[indices],
        frame["blue_goal_in_next_ten_seconds"].to_numpy()[indices],
        frame["orange_goal_in_next_ten_seconds"].to_numpy()[indices],
        frame["neither_goal_in_next_ten_seconds"].to_numpy()[indices],
        targets_per_team=targets_per_team, geometric_width=geometric_width)
    
    return result


def output_count(targets_per_team):
    if targets_per_team < 1:
        raise ValueError("targets_per_team must be at least one")
    return 2 * int(targets_per_team) + 1

def bucket_edges(targets_per_team=10, geometric_width=1.25):
    if targets_per_team < 1 or not math.isfinite(geometric_width) or geometric_width <= 0:
        raise ValueError("targets_per_team must be positive and geometric_width finite and positive")
    raw = np.power(float(geometric_width), np.arange(targets_per_team, dtype=np.float64))
    edges = np.empty(targets_per_team + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:] = np.cumsum(raw / raw.sum() * 10.0)
    edges[-1] = 10.0
    return edges

def temporal_targets(
    time_until_next_goal, 
    blue_goal, 
    orange_goal, 
    neither_goal, 
    *, 
    targets_per_team=10, 
    geometric_width=1.25,     
    sigma_base=1e-5, 
    sigma_growth=0.15
):
    dt = np.asarray(time_until_next_goal, dtype=np.float64)
    blue, orange, neither = (np.asarray(value, dtype=bool) for value in (blue_goal, orange_goal, neither_goal))
    
    if not (dt.shape == blue.shape == orange.shape == neither.shape) or np.any(blue.astype(np.int8) + orange.astype(np.int8) + neither.astype(np.int8) != 1):
        raise ValueError("goal state columns must be aligned and exactly one-hot")
    
    teams = np.where(blue, 0, np.where(orange, 1, -1)).astype(np.int64)
    edges = bucket_edges(targets_per_team, geometric_width)
    values = np.zeros((dt.size, output_count(targets_per_team)), dtype=np.float32)
    values[:, -1] = 1.0
    qualifying = np.isfinite(dt) & (dt > 0.0) & (dt <= 10.0) & np.isin(teams, (0, 1))
    collapsed = np.where(qualifying, teams, 2).astype(np.int64)
    rows = np.flatnonzero(qualifying)
    
    if rows.size:
        mu = torch.from_numpy(dt[rows]).to(torch.float64)[:, None]
        sigma = float(sigma_base) + float(sigma_growth) * mu
        edge_tensor = torch.from_numpy(edges).to(torch.float64)[None, :]
        cdf = 0.5 * (1.0 + torch.erf((edge_tensor - mu) / (sigma * math.sqrt(2.0))))
        mass = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0).numpy().astype(np.float32)
        valid_values = np.zeros((rows.size, output_count(targets_per_team)), dtype=np.float32)
        
        for team in (0, 1):
            selected = teams[rows] == team
            valid_values[selected, team * targets_per_team:(team + 1) * targets_per_team] = mass[selected]
            
        valid_values[:, -1] = np.clip(1.0 - mass.sum(axis=1), 0.0, 1.0)
        values[rows] = valid_values
        
    targets = torch.from_numpy(values)
    
    if not torch.allclose(targets.sum(dim=-1), torch.ones(dt.size), atol=1e-5):
        raise RuntimeError("temporal targets must sum to one")
        
    return targets, torch.from_numpy(collapsed)


def collapse_probabilities(probabilities, targets_per_team):
    if probabilities.shape[-1] != output_count(targets_per_team):
        raise ValueError("probability width does not match targets_per_team")
    return torch.stack((probabilities[..., :targets_per_team].sum(dim=-1),
                        probabilities[..., targets_per_team:2 * targets_per_team].sum(dim=-1),
                        probabilities[..., -1]), dim=-1)

