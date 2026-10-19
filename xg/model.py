# Train replay-grouped Rocket League expected-goals models.

from __future__ import annotations

import gc
import json
import joblib
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from pathlib import Path
from typing import Any, Mapping, Sequence
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from .config import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    MODEL_FOLDER,
    MODEL_SEGMENTS,
    XGModelConfig,
    XGRuntimeConfig,
    XGTrainConfig,
)
from .dataset import (
    FRAMES_FOLDER,
    RANDOM_STATE,
    _feature_column_class,
    _feature_column_classes,
    _normalize_segments,
    _sample_feature_matrix,
    _variable_feature_columns,
    load_segment_training_frames,
    load_training_parquet,
    prepare_model_data,
    to_sparse_matrix,
)

MODEL_FOLDER = str(MODEL_FOLDER)

DEFAULT_PARAMS = {
    'n_estimators': 400,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 2.0,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
}


optuna.logging.set_verbosity(optuna.logging.WARNING)
# ANALYZERL XG
# Trains replay-grouped expected-goals models from labelled frame data.


class AnalyzeRLxG:
    # Gradient-boosted expected-goals model backed by XGBoost.

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        random_state: int = RANDOM_STATE,
        jobs: int | None = None,
        gpu: bool = False,
    ) -> None:
        # Initialize a binary XGBoost classifier for shot-goal labels.
        model_params = {**DEFAULT_PARAMS, **dict(params or {})}
        model_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cuda' if gpu else 'cpu',
            'random_state': random_state,
            'n_jobs': jobs,
        })
        self.params = model_params
        self.model = xgb.XGBClassifier(**model_params)

    def fit(
        self,
        matrix: sparse.csr_matrix,
        labels: np.ndarray,
        valid_matrix: sparse.csr_matrix | None = None,
        valid_labels: np.ndarray | None = None,
        early_stopping_rounds: int = 50,
    ) -> AnalyzeRLxG:
        # Validation folds own early stopping; the final refit uses the selected tree count.
        if valid_matrix is not None and valid_labels is not None:
            self.model.set_params(early_stopping_rounds=int(early_stopping_rounds))
            self.model.fit(
                matrix,
                labels,
                eval_set=[(valid_matrix, valid_labels)],
                verbose=False,
            )
        else:
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(matrix, labels, verbose=False)
        return self

    def predict_proba(self, matrix: sparse.csr_matrix) -> np.ndarray:
        # Return class probabilities for a prepared sparse feature matrix.
        return self.model.predict_proba(matrix)

    @property
    def feature_importances_(self) -> np.ndarray:
        # Expose fitted XGBoost feature importances.
        return self.model.feature_importances_

    @property
    def best_iteration_(self) -> int:
        # Return the early-stopped tree count or configured estimator count.
        value = getattr(self.model, 'best_iteration', None)
        return int(value) + 1 if value is not None else int(self.params['n_estimators'])


def _grouped_split(labels: np.ndarray, groups: np.ndarray, train_pct: float, seed: int):
    # Replay groups cannot cross the train/test boundary without leaking game state.
    rows = np.arange(len(labels))
    splitter = GroupShuffleSplit(n_splits=50, train_size=train_pct, random_state=seed)
    for train_idx, test_idx in splitter.split(rows, labels, groups):
        if len(np.unique(labels[train_idx])) == 2 and len(np.unique(labels[test_idx])) == 2:
            return train_idx, test_idx
    raise ValueError('Could not create a replay-grouped train/test split containing both outcomes')


def _suggest_params(trial: optuna.Trial, fixed: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(fixed)
    suggestions = {
        'n_estimators': lambda: trial.suggest_int('n_estimators', 300, 3000, log=True),
        'max_depth': lambda: trial.suggest_int('max_depth', 2, 12),
        'max_leaves': lambda: trial.suggest_int('max_leaves', 8, 256, log=True),
        'grow_policy': lambda: trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'learning_rate': lambda: trial.suggest_float('learning_rate', 0.003, 0.3, log=True),
        'subsample': lambda: trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': lambda: trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel': lambda: trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'colsample_bynode': lambda: trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'min_child_weight': lambda: trial.suggest_float('min_child_weight', 0.1, 50.0, log=True),
        'gamma': lambda: trial.suggest_float('gamma', 0.0, 10.0),
        'max_delta_step': lambda: trial.suggest_float('max_delta_step', 0.0, 10.0),
        'reg_lambda': lambda: trial.suggest_float('reg_lambda', 1e-4, 100.0, log=True),
        'reg_alpha': lambda: trial.suggest_float('reg_alpha', 1e-5, 50.0, log=True),
        'max_bin': lambda: trial.suggest_categorical('max_bin', [128, 256, 384, 512, 768]),
    }
    for name, suggest in suggestions.items():
        if name not in params:
            params[name] = suggest()
    return params


def _study_results(study: optuna.Study) -> pl.DataFrame:
    records = []
    for trial in study.trials:
        records.append({
            'iteration': trial.number,
            'mean_log_loss': trial.value,
            'std_log_loss': trial.user_attrs.get('std_log_loss'),
            'mean_best_iteration': trial.user_attrs.get('mean_best_iteration'),
            'completed_folds': trial.user_attrs.get('completed_folds', 0),
            'state': trial.state.name,
            'params': trial.user_attrs.get('params', json.dumps(trial.params, sort_keys=True)),
        })
    return pl.DataFrame(records) if records else pl.DataFrame({
        'iteration': [], 'mean_log_loss': [], 'std_log_loss': [],
        'mean_best_iteration': [], 'completed_folds': [], 'state': [], 'params': [],
    })


def _cross_validate(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    groups: np.ndarray,
    cv: int,
    n_iters: int,
    params: Mapping[str, Any] | None,
    seed: int,
    jobs: int | None,
    gpu: bool,
    early_stopping_rounds: int,
    cv_path: Path,
    segment: str,
) -> tuple[dict[str, Any], pl.DataFrame]:
    splits = min(cv, len(np.unique(groups)))
    if splits < 2:
        raise ValueError('Cross-validation requires at least two replay groups')
    folds = list(
        StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
        .split(np.arange(len(labels)), labels, groups)
    )
    if any(len(np.unique(labels[train_idx])) < 2 for train_idx, _ in folds):
        raise ValueError('A cross-validation training fold contains only one outcome')
    fixed_params = dict(params or {})

    def evaluate(candidate, iteration, trial=None):
        # Each validation fold owns early stopping and contributes its best tree count.
        losses = []
        best_iterations = []
        for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
            print(
                f'[{segment}] trial {iteration + 1}: fold {fold_number}/{splits}',
                flush=True,
            )
            fold_train = matrix[train_idx]
            fold_valid = matrix[valid_idx]
            model = AnalyzeRLxG(candidate, seed + iteration, jobs, gpu).fit(
                fold_train,
                labels[train_idx],
                fold_valid,
                labels[valid_idx],
                early_stopping_rounds,
            )
            predictions = model.predict_proba(fold_valid)[:, 1]
            losses.append(log_loss(labels[valid_idx], predictions, labels=[0, 1]))
            best_iterations.append(model.best_iteration_)
            del model, fold_train, fold_valid, predictions
            if trial is not None:
                trial.set_user_attr('completed_folds', fold_number)
                trial.report(float(np.mean(losses)), step=fold_number)
                if trial.should_prune():
                    trial.set_user_attr('std_log_loss', float(np.std(losses)))
                    trial.set_user_attr('mean_best_iteration', int(np.mean(best_iterations)))
                    raise optuna.TrialPruned()
        return losses, best_iterations

    if n_iters == 0:
        candidate = {**DEFAULT_PARAMS, **fixed_params}
        losses, best_iterations = evaluate(candidate, 0)
        candidate['n_estimators'] = max(1, int(np.mean(best_iterations)))
        results = pl.DataFrame([{
            'iteration': 0,
            'mean_log_loss': float(np.mean(losses)),
            'std_log_loss': float(np.std(losses)),
            'mean_best_iteration': candidate['n_estimators'],
            'completed_folds': splits,
            'state': 'COMPLETE',
            'params': json.dumps(candidate, sort_keys=True),
        }])
        results.write_csv(cv_path)
        print(f'[{segment}] CV checkpoint written: {cv_path}', flush=True)
        return candidate, results

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(5, n_iters),
        n_warmup_steps=1,
    )
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)

    def objective(trial):
        candidate = _suggest_params(trial, fixed_params)
        trial.set_user_attr('params', json.dumps(candidate, sort_keys=True))
        losses, best_iterations = evaluate(candidate, trial.number, trial)
        trial.set_user_attr('std_log_loss', float(np.std(losses)))
        trial.set_user_attr('mean_best_iteration', int(np.mean(best_iterations)))
        trial.set_user_attr('completed_folds', splits)
        return float(np.mean(losses))

    def checkpoint(study, trial):
        # Persist every trial so an interrupted search retains completed CV work.
        results = _study_results(study)
        results.write_csv(cv_path)
        value = 'pruned' if trial.value is None else f'{trial.value:.6f}'
        print(
            f'[{segment}] trial {trial.number + 1}/{n_iters} {trial.state.name.lower()} '
            f'(log loss: {value}); checkpoint written',
            flush=True,
        )

    study.optimize(
        objective,
        n_trials=n_iters,
        callbacks=[checkpoint],
        gc_after_trial=True,
        show_progress_bar=False,
    )
    best_trial = study.best_trial
    best_params = {**best_trial.params, **fixed_params}
    best_params['n_estimators'] = max(
        1,
        int(best_trial.user_attrs.get('mean_best_iteration', best_params['n_estimators'])),
    )
    return best_params, _study_results(study)


def _write_evaluation(
    output: Path,
    segment: str,
    model: AnalyzeRLxG,
    feature_columns: Sequence[str],
    shap_matrix: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
):
    import matplotlib.pyplot as plt

    import shap

    try:
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(shap_matrix)
    except ValueError as error:
        print(f'[{segment}] TreeExplainer unavailable; using KernelExplainer ({error})', flush=True)
        background = shap_matrix[:min(10, len(shap_matrix))]
        explainer = shap.KernelExplainer(
            lambda values: model.model.predict_proba(values)[:, 1],
            background,
        )
        shap_values = explainer.shap_values(
            shap_matrix,
            nsamples=min(200, shap_matrix.shape[1] * 2 + 1),
            silent=True,
        )
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values, dtype=np.float32)
    np.save(output / f'shap_values_{segment}.npy', shap_values)
    mean_abs_shap = np.nanmean(np.abs(shap_values), axis=0)

    model_importance = pl.DataFrame({
        'feature': feature_columns,
        'column_class': [_feature_column_class(column) for column in feature_columns],
        'mean_abs_shap': mean_abs_shap,
        'importance': model.feature_importances_,
    }).sort('mean_abs_shap', descending=True)
    _write_csv_when_unlocked(model_importance, output / f'feature_importance_{segment}.csv')
    top_model_importance = model_importance.head(20).sort('mean_abs_shap')
    plt.figure(figsize=(10, 7))
    plt.barh(
        top_model_importance['feature'].to_list(),
        top_model_importance['mean_abs_shap'].to_numpy(),
        color='#2f6f73',
    )
    plt.xlabel('Mean absolute SHAP value')
    plt.title(f'AnalyzeRL xG Feature Importance - {segment}')
    plt.tight_layout()
    plt.savefig(output / f'feature_importance_{segment}.png', dpi=160)
    plt.close()

    shap.summary_plot(
        shap_values,
        shap_matrix,
        feature_names=list(feature_columns),
        max_display=25,
        show=False,
    )
    plt.gcf().set_size_inches(10, 8)
    plt.title(f'AnalyzeRL xG SHAP Summary - {segment}')
    plt.tight_layout()
    plt.savefig(output / f'feature_importance_summary_{segment}.png', dpi=160)
    plt.close()

    top_feature = model_importance['feature'][0]
    shap.dependence_plot(
        top_feature,
        shap_values,
        shap_matrix,
        feature_names=list(feature_columns),
        interaction_index='auto',
        show=False,
    )
    plt.gcf().set_size_inches(9, 7)
    plt.title(f'AnalyzeRL xG SHAP Dependence - {segment}: {top_feature}')
    plt.tight_layout()
    plt.savefig(output / f'feature_importance_dependence_{segment}.png', dpi=160)
    plt.close()

    false_positive, true_positive, _ = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    plt.figure(figsize=(8, 7))
    plt.plot(false_positive, true_positive, label=f'AUC = {auc:.4f}', color='#2f6f73')
    plt.plot([0, 1], [0, 1], '--', color='#777777')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'AnalyzeRL xG ROC - {segment}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f'roc_auc_{segment}.png', dpi=160)
    plt.close()

    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(np.digitize(predictions, bins) - 1, 0, len(bins) - 2)
    predicted, observed = [], []
    for bin_id in range(len(bins) - 1):
        mask = bin_ids == bin_id
        if np.any(mask):
            predicted.append(float(np.mean(predictions[mask])))
            observed.append(float(np.mean(labels[mask])))
    plt.figure(figsize=(8, 7))
    plt.plot(predicted, observed, 's-', color='#2f6f73', label=segment)
    plt.plot([0, 1], [0, 1], '--', color='#777777', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed goal rate')
    plt.title(f'AnalyzeRL xG Calibration - {segment}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f'calibration_{segment}.png', dpi=160)
    plt.close()


def _write_csv_when_unlocked(frame: pl.DataFrame, path: Path) -> None:
    # Write a CSV, tolerating locked generated artifacts on Windows.
    try:
        frame.write_csv(path)
    except OSError:
        if not path.exists():
            raise


def _train_segment(
    frames: pl.DataFrame,
    segment: str,
    directory: str | Path,
    cv: int,
    n_iters: int,
    train_test_split_pct: float,
    params: Mapping[str, Any] | None,
    seed: int,
    jobs: int | None,
    gpu: bool,
    early_stopping_rounds: int,
    prepared: bool = False,
    include_prior_event_history: bool = False,
    test_frames: pl.DataFrame | None = None,
) -> dict[str, Any]:
    print(f'[{segment}] preparing model features from {frames.height:,} shots', flush=True)
    features = frames if prepared else prepare_model_data(
        frames,
        include_prior_event_history=include_prior_event_history,
    )
    labels = features['is_goal'].to_numpy().astype(np.int8)
    groups = features['replay_id'].to_numpy()
    if features.height < 20 or len(np.unique(labels)) < 2:
        raise ValueError(f'{segment} does not contain enough shot outcomes to train')
    test_features = (
        None if test_frames is None
        else test_frames if prepared
        else prepare_model_data(test_frames, include_prior_event_history=include_prior_event_history)
    )
    if test_features is None:
        train_idx, test_idx = _grouped_split(labels, groups, train_test_split_pct, seed)
    else:
        train_idx = np.arange(features.height)
        test_idx = np.arange(test_features.height)
        overlap = set(groups).intersection(test_features['replay_id'].to_list())
        if overlap:
            raise ValueError(f'{segment} train/test replay leakage: {len(overlap)} overlapping replays')
    candidate_columns = [
        column for column in features.columns
        if column not in {'replay_id', 'event_id', 'is_goal', 'model_segment'}
    ]
    feature_columns, removed_constant_features = _variable_feature_columns(
        features[train_idx],
        candidate_columns,
    )
    print(
        f'[{segment}] retained {len(feature_columns):,}/{len(candidate_columns):,} features; '
        f'removed {len(removed_constant_features):,} single-valued columns',
        flush=True,
    )
    print(f'[{segment}] building sparse train/test matrices', flush=True)
    train_matrix = to_sparse_matrix(features[train_idx], feature_columns)
    test_matrix = to_sparse_matrix(
        features[test_idx] if test_features is None else test_features,
        feature_columns,
    )
    output = Path(directory) / segment
    output.mkdir(parents=True, exist_ok=True)
    cv_path = output / f'cross_validation_{segment}.csv'
    print(
        f'[{segment}] starting {cv}-fold CV with '
        f'{max(1, n_iters)} trial(s) on {"GPU" if gpu else "CPU"}',
        flush=True,
    )
    best_params, cv_results = _cross_validate(
        train_matrix,
        labels[train_idx],
        groups[train_idx],
        cv,
        n_iters,
        params,
        seed,
        jobs,
        gpu,
        early_stopping_rounds,
        cv_path,
        segment,
    )
    print(
        f'[{segment}] fitting final model with {best_params["n_estimators"]} trees',
        flush=True,
    )
    model = AnalyzeRLxG(best_params, seed, jobs, gpu).fit(train_matrix, labels[train_idx])
    predictions = model.predict_proba(test_matrix)[:, 1]
    test_eval_features = features[test_idx] if test_features is None else test_features
    test_labels = (
        labels[test_idx] if test_features is None
        else test_features['is_goal'].to_numpy().astype(np.int8)
    )
    test_groups = (
        groups[test_idx] if test_features is None
        else test_features['replay_id'].to_numpy()
    )
    shap_matrix = _sample_feature_matrix(test_eval_features, feature_columns, seed=seed)
    column_classes = _feature_column_classes(feature_columns)
    cv_results.write_csv(cv_path)
    _write_evaluation(
        output,
        segment,
        model,
        feature_columns,
        shap_matrix,
        test_labels,
        predictions,
    )
    metrics = {
        'segment': segment,
        'n_replays': len(np.unique(np.concatenate([groups, test_groups]))),
        'n_events': features.height + (0 if test_features is None else test_features.height),
        'n_goals': int(labels.sum() + (0 if test_features is None else test_labels.sum())),
        'n_train_goals': int(labels.sum()),
        'n_test_goals': int(test_labels.sum()),
        'n_train_events': len(train_idx),
        'n_test_events': len(test_idx) if test_features is None else test_features.height,
        'n_train_replays': len(np.unique(groups[train_idx])),
        'n_test_replays': len(np.unique(test_groups)),
        'n_features': len(feature_columns),
        'feature_class_counts': {
            column_class: len(columns)
            for column_class, columns in column_classes.items()
        },
        'removed_constant_features': removed_constant_features,
        'train_test_split_pct': train_test_split_pct,
        'test_source_holdout': test_frames is not None,
        'cv_folds': cv,
        'search_iterations': n_iters,
        'early_stopping_rounds': early_stopping_rounds,
        'test_log_loss': float(log_loss(test_labels, predictions, labels=[0, 1])),
        'test_brier': float(brier_score_loss(test_labels, predictions)),
        'test_roc_auc': float(roc_auc_score(test_labels, predictions)),
        'test_average_precision': float(average_precision_score(test_labels, predictions)),
        'best_params': {**DEFAULT_PARAMS, **best_params},
    }
    with (output / f'metrics_{segment}.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    joblib.dump({
        'model': model,
        'feature_columns': feature_columns,
        'column_classes': column_classes,
        'model_segment': segment,
    }, output / f'xg_model_{segment}.joblib')
    print(f'[{segment}] model and evaluation artifacts written to {output}', flush=True)
    return metrics


def analyzerl_xg(
    frames_folder: str | Path = FRAMES_FOLDER,
    training_parquet: str | Path | None = None,
    directory: str | Path = MODEL_FOLDER,
    model_segments: Sequence[str] | str | None = MODEL_SEGMENTS,
    limit: int | None = None,
    cv: int = 5,
    n_iters: int = 25,
    train_test_split_pct: float = 0.8,
    jobs: int | None = None,
    params: Mapping[str, Any] | None = None,
    random_state: int = RANDOM_STATE,
    gpu: bool | None = None,
    include_event_context: bool = False,
    early_stopping_rounds: int = 50,
    train_config: XGTrainConfig | None = None,
    model_config: XGModelConfig | None = None,
    runtime_config: XGRuntimeConfig | None = None,
    test_frames_folder: str | Path | None = None,
    train_replay_ids: Sequence[str] | None = None,
    test_replay_ids: Sequence[str] | None = None,
    train_frame_paths: Sequence[str | Path] | Mapping[str, Sequence[str | Path]] | None = None,
    test_frame_paths: Sequence[str | Path] | Mapping[str, Sequence[str | Path]] | None = None,
) -> pl.DataFrame:
    # Train and evaluate an xG model for each requested replay segment.
    if train_config is not None:
        frames_folder = train_config.frames_folder
        test_frames_folder = train_config.test_frames_folder
        training_parquet = train_config.training_parquet
        directory = train_config.output_dir
        model_segments = train_config.model_segments
        limit = train_config.replay_limit
        cv = train_config.cv_folds
        n_iters = train_config.search_iterations
        train_test_split_pct = train_config.train_test_split_pct
        early_stopping_rounds = train_config.early_stopping_rounds
        include_event_context = train_config.include_prior_event_history
    if model_config is not None:
        if model_config.model_params is not None:
            params = model_config.model_params
        random_state = model_config.random_state
    if runtime_config is not None:
        jobs = runtime_config.jobs
        gpu = runtime_config.gpu
    if gpu is None:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
    segments = _normalize_segments(model_segments)
    if not 0.0 < train_test_split_pct < 1.0:
        raise ValueError('train_test_split_pct must be strictly between 0 and 1')
    if cv < 2:
        raise ValueError('cv must be at least 2')
    if n_iters < 0:
        raise ValueError('n_iters cannot be negative')
    if early_stopping_rounds <= 0:
        raise ValueError('early_stopping_rounds must be a positive integer')
    results = []
    for segment in segments:
        # Load, train, and release one mode before scanning data for the next mode.
        print(f'[{segment}] loading training replays', flush=True)
        prepared = training_parquet is not None
        segment_train_paths = (
            train_frame_paths.get(segment)
            if isinstance(train_frame_paths, Mapping)
            else train_frame_paths
        )
        segment_test_paths = (
            test_frame_paths.get(segment)
            if isinstance(test_frame_paths, Mapping)
            else test_frame_paths
        )
        if prepared:
            segment_frames = load_training_parquet(
                training_parquet,
                segment,
                limit,
                random_state,
            )
        else:
            segment_frames = load_segment_training_frames(
                frames_folder,
                segment,
                limit,
                random_state,
                include_event_context,
                frame_paths=None if segment_train_paths is None else [Path(path) for path in segment_train_paths],
                selected_replay_ids=train_replay_ids,
            )
        test_segment_frames = None
        if test_frames_folder is not None:
            test_segment_frames = load_segment_training_frames(
                test_frames_folder,
                segment,
                limit,
                random_state,
                include_event_context,
                frame_paths=None if segment_test_paths is None else [Path(path) for path in segment_test_paths],
                selected_replay_ids=test_replay_ids,
            )
        if segment_frames.is_empty():
            print(f'[{segment}] no matching shot events; skipping', flush=True)
            continue
        if test_segment_frames is not None and test_segment_frames.is_empty():
            raise ValueError(f'{segment} has no shot events in the configured LGPS test source')
        try:
            results.append(_train_segment(
                segment_frames,
                segment,
                directory,
                cv,
                n_iters,
                train_test_split_pct,
                params,
                random_state,
                jobs,
                gpu,
                early_stopping_rounds,
                prepared,
                include_event_context,
                test_segment_frames,
            ))
        finally:
            del segment_frames
            if test_segment_frames is not None:
                del test_segment_frames
            gc.collect()
            print(f'[{segment}] released segment training memory', flush=True)
    if not results:
        raise ValueError('No shot events matched the requested model_segments')
    return pl.DataFrame(results)


if __name__ == '__main__':
    analyzerl_xg(
        n_iters=20,
        model_segments=MODEL_SEGMENTS,
        include_event_context=True,
        jobs=2
    )
