from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

FRAMES_FOLDER = Path("../data/frames/rlcs/x")
TEST_FRAMES_FOLDER = Path("../data/frames/rlcs/2026")
MODEL_FOLDER = Path("models/xg")
MODEL_SEGMENTS = ("duels", "doubles", "standard")

@dataclass(frozen=True)
class XGModelConfig:
    random_state: int = 42
    model_params: Mapping[str, Any] | None = None

@dataclass(frozen=True)
class XGTrainConfig:
    frames_folder: Path = FRAMES_FOLDER
    test_frames_folder: Path | None = TEST_FRAMES_FOLDER
    training_parquet: Path | None = None
    output_dir: Path = MODEL_FOLDER
    model_segments: tuple[str, ...] = MODEL_SEGMENTS
    replay_limit: int | None = None
    cv_folds: int = 5
    search_iterations: int = 30
    train_test_split_pct: float = 0.8
    early_stopping_rounds: int = 50
    include_prior_event_history: bool = False

@dataclass(frozen=True)
class XGRuntimeConfig:
    jobs: int | None = None
    gpu: bool | None = None

DEFAULT_MODEL_CONFIG = XGModelConfig()
DEFAULT_TRAIN_CONFIG = XGTrainConfig()
DEFAULT_RUNTIME_CONFIG = XGRuntimeConfig()
