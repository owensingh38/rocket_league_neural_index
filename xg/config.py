from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from shared import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
FRAMES_FOLDER = ROOT / DATA_DIR / "frames" / "x"
TEST_FRAMES_FOLDER = ROOT / DATA_DIR / "frames" / "2026"
MODEL_FOLDER = ROOT / "results" / "xg"
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

@dataclass(frozen=True)
class XGRuntimeConfig:
    jobs: int | None = None
    gpu: bool | None = None
    # Distribute XGBoost CV and final fits through xgboost.dask.
    dask: bool = False
    dask_scheduler_address: str | None = None
    dask_workers: int | None = None
    dask_chunk_rows: int = 65536

DEFAULT_MODEL_CONFIG = XGModelConfig()
DEFAULT_TRAIN_CONFIG = XGTrainConfig()
DEFAULT_RUNTIME_CONFIG = XGRuntimeConfig()
