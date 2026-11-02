from dataclasses import dataclass, field, replace
from pathlib import Path
from props import DATA_DIR, INPUT_DIR
from .camera import PlayerCameraViewEncoder
from .targets import output_count

OUTPUT_DIR = DATA_DIR / "results" / "lgps"

@dataclass(frozen=True)
class LGPSModelConfig:
    d_model: int = 128
    num_heads: int = 8
    num_blocks: int = 4
    loops_per_block: int = 2
    ff_dim: int = 256
    expanded_geometry: bool = False
    pairwise_rank: int = 32
    dropout: float = 0.05
    targets_per_team: int = 61
    geometric_width: float = 1.0
    attention_mode: str = "relational"
    horizon_seconds: float | None = None
    horizon_max: float = 30.0
    sample_rate: float = 10.0
    model_entities: str | tuple[str, ...] = "all"
    # Train goal-only outputs: blue and orange scoring buckets, with no
    # neither class or output bucket.
    two_way_prediction: bool = True

    @property
    def num_outputs(self) -> int:
        return output_count(
            self.targets_per_team, self.horizon_seconds, self.two_way_prediction,
        )

@dataclass(frozen=True)
class LGPSTrainConfig:
    input_path: Path = INPUT_DIR
    test_path: Path | None = None
    batch_size: int = 8192
    validation_batch_size: int = 8192
    effective_batch_size: int = 65536
    epochs: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    gradient_clip_norm: float = 1.0
    seed: int = 42
    activation_checkpointing: bool = False
    train_split: float = 0.95
    val_split: float = 0.05
    log_interval: int = 100
    # Exclude stints with no qualifying next goal from training/evaluation data.
    drop_neither: bool = False
    player_view: PlayerCameraViewEncoder.Settings | bool | None = field(
        default_factory=PlayerCameraViewEncoder.Settings,
    )

def normalize_lgps_train_config(config):
    # Normalize the camera shorthand where the config enters training.
    camera = config.player_view
    if camera is True:
        camera = PlayerCameraViewEncoder.Settings()
    elif camera is False:
        camera = None
    elif camera is not None and not isinstance(camera, PlayerCameraViewEncoder.Settings):
        raise TypeError('player_view must be a camera encoder Settings object, True, or None')
    if camera is config.player_view:
        return config
    return replace(config, player_view=camera)

@dataclass(frozen=True)
class LGPSRuntimeConfig:
    device: str = "cuda"
    world_size: int = 2
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
    two_way_prediction=False,
    targets_per_team=14,
    geometric_width=1.5,
    horizon_seconds=None,
    horizon_max=30.0,
    sample_rate=10.0,
    model_entities=["players", "ball", "goals", "big_boosts", "context"]
)

DEFAULT_TRAIN_CONFIG = LGPSTrainConfig(
    input_path=INPUT_DIR,
    activation_checkpointing=False,
    drop_neither=False,
    batch_size=8192,
    validation_batch_size=8192,
    effective_batch_size=65536,
    epochs=10,
    learning_rate=0.0003,
    min_learning_rate=0.000001,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
    train_split=0.95,
    val_split=0.05,
    log_interval=100,
    player_view=None
)

DEFAULT_RUNTIME_CONFIG = LGPSRuntimeConfig(
    device="cuda",
    world_size=2,
    ddp_backend="nccl",
    find_unused_parameters=False,
    amp_policy="auto",
    per_gpu_batch_size=8192,
    data_workers=2,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
    compile_mode=None,
    wandb_project="latent-goal-probability-state",
    wandb_entity="project-signal",
    wandb_run_name="LGPSTransformer",
    wandb_mode="online"
)
