from dataclasses import dataclass
from pathlib import Path
from props import DATA_DIR, INPUT_DIR, normalize_model_entities
from .aux_tasks import LGPSBallStateAux, LGPSNextTouchAux, LGPSPlayerStateAux

OUTPUT_DIR = DATA_DIR / "results" / "lgps"

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
    sample_rate: float = 10.0
    model_entities: str | tuple[str, ...] = "all"

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
    gradnorm_alpha: float = 1.5
    gradnorm_momentum: float = 0.9
    future_frames: tuple[int, ...] | None = None
    aux_tasks: tuple[type, ...] | list[type] | None = None

    def __post_init__(self):
        # Task classes are selected directly from aux.py. An empty selection
        # trains the goal head alone and avoids building future-state targets.
        tasks = () if self.aux_tasks is None else tuple(self.aux_tasks)
        object.__setattr__(self, "aux_tasks", tasks)
        
        frames = None if self.future_frames is None else tuple(self.future_frames)
        object.__setattr__(self, "future_frames", frames)

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

DEFAULT_MODEL_CONFIG = LGPSModelConfig(
    d_model=128,
    num_heads=8,
    num_blocks=4,
    loops_per_block=2,
    ff_dim=256,
    expanded_geometry=True,
    pairwise_rank=32,
    dropout=0.05,
    targets_per_team=10,
    geometric_width=1.5,
    sample_rate=10.0,
    model_entities=["players", "ball", "goals", "big_boosts", "context"]
)

DEFAULT_TRAIN_CONFIG = LGPSTrainConfig(
    input_path=INPUT_DIR,
    batch_size=2048,
    validation_batch_size=8192,
    effective_batch_size=16384,
    epochs=10,
    learning_rate=0.0003,
    min_learning_rate=0.000001,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
    train_split=0.95,
    val_split=0.05,
    log_interval=100,
    future_frames=(10, 30, 60, 100),
    aux_tasks=[
        LGPSBallStateAux,
        LGPSPlayerStateAux,
        LGPSNextTouchAux
    ],
    gradnorm_alpha=1.5,
    gradnorm_momentum=0.9
)

DEFAULT_RUNTIME_CONFIG = LGPSRuntimeConfig(
    device="cuda",
    world_size=1,
    ddp_backend="nccl",
    find_unused_parameters=False,
    amp_policy="auto",
    per_gpu_batch_size=2048,
    data_workers=2,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
    compile_mode="default",
    activation_checkpointing=False,
    wandb_project=None,
    wandb_entity=None,
    wandb_run_name=None,
    wandb_mode=None,
)
