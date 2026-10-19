from __future__ import annotations

import joblib
import polars as pl
from pathlib import Path
from xg.dataset import prepare_model_data, to_sparse_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "xg" / "standard" / "xg_model_standard.joblib"
OUTPUT_DIR = ROOT / "inference" / "output" / "xg"
PROBABILITY_COLUMN = "xg"
RAW_REQUIRED = {"shooter_team", "ball_pos_x", "ball_pos_y", "ball_pos_z",
                "ball_vel_x", "ball_vel_y", "ball_vel_z"}

def _input_paths(path: Path, output_dir: Path) -> list[Path]:
    if path.is_file():
        files = [path] if path.suffix.lower() in {".csv", ".parquet"} else []
    elif path.is_dir():
        files = sorted(file for file in path.rglob("*")
                       if file.is_file() and file.suffix.lower() in {".csv", ".parquet"}
                       and not file.resolve().is_relative_to(output_dir))
    else:
        raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet files found under {path}")
    return files


def apply_xg(path: str | Path, model_path: str | Path = MODEL_PATH,
             output_dir: str | Path = OUTPUT_DIR) -> list[Path]:

    path = Path(path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    
    artifact = joblib.load(model_path)
    columns = artifact["feature_columns"]
    model = artifact["model"]

    outputs = []
    for source in _input_paths(path, output_dir):
        frame = (pl.read_csv(source, infer_schema_length=None) if source.suffix.lower() == ".csv"
                 else pl.read_parquet(source))
        
        if PROBABILITY_COLUMN in frame.columns:
            raise ValueError(f"{source}: '{PROBABILITY_COLUMN}' already exists")
        
        if not frame.is_empty():
            if set(columns).issubset(frame.columns):
                features = frame
            else:
                missing_raw = RAW_REQUIRED - set(frame.columns)
                if missing_raw:
                    raise ValueError(f"{source}: missing model features and shot columns: "
                                     f"{sorted(missing_raw)}")
                features = prepare_model_data(frame)
                missing_features = set(columns) - set(features.columns)
                if missing_features:
                    raise ValueError(f"{source}: model features unavailable after preparation: "
                                     f"{sorted(missing_features)}")
            probabilities = model.predict_proba(to_sparse_matrix(features, columns))[:, 1]
            frame = frame.with_columns(pl.Series(PROBABILITY_COLUMN, probabilities))
        else:
            frame = frame.with_columns(pl.Series(PROBABILITY_COLUMN, [], dtype=pl.Float32))

        destination = output_dir / (source.relative_to(path) if path.is_dir() else source.name)
        
        if destination == source:
            raise ValueError(f"Output would overwrite input: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if destination.suffix.lower() == ".csv":
            frame.write_csv(destination)
        else:
            frame.write_parquet(destination)
        
        outputs.append(destination)
    
    return outputs
