# Train replay-grouped Rocket League expected-goals models.

from __future__ import annotations

import gc
import json
import joblib
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence
from scipy import sparse
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .config import (
    MODEL_FOLDER,
    MODEL_SEGMENTS,
    XGModelConfig,
    XGRuntimeConfig,
    XGTrainConfig,
)
from .dataset import (
    FRAMES_FOLDER,
    RANDOM_STATE,
    _feature_column_classes,
    _normalize_segments,
    _feature_matrix,
    _variable_feature_columns,
    load_segment_training_frames,
    load_training_parquet,
    prepare_model_data,
    to_sparse_matrix,
)

MODEL_FOLDER = str(MODEL_FOLDER)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        dask_client=None,
        dask_chunk_rows: int = 65536,
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
        self.dask_client = dask_client
        self.dask_chunk_rows = int(dask_chunk_rows)

    def fit(
        self,
        matrix: sparse.csr_matrix,
        labels: np.ndarray,
        valid_matrix: sparse.csr_matrix | None = None,
        valid_labels: np.ndarray | None = None,
        early_stopping_rounds: int = 50,
    ) -> AnalyzeRLxG:
        # Validation folds own early stopping; the final refit uses the selected tree count.
        if self.dask_client is not None:
            import dask.array as da
            from dask import delayed
            from xgboost import dask as dxgb

            def distributed_array(values, *, labels=False):
                rows = values.shape[0]
                workers = sorted(self.dask_client.scheduler_info()['workers'])
                worker_count = max(1, len(workers))
                target_rows_per_worker = max(
                    1, (rows + worker_count - 1) // worker_count,
                )
                chunk_rows = min(
                    max(1, self.dask_chunk_rows),
                    target_rows_per_worker,
                )
                blocks = []

                def keep_block(block):
                    return block

                for block_number, start in enumerate(range(0, rows, chunk_rows)):
                    end = min(start + chunk_rows, rows)
                    block = values[start:end]
                    worker = workers[block_number % worker_count]
                    future = self.dask_client.scatter(
                        block,
                        workers=[worker],
                        broadcast=False,
                        hash=False,
                    )
                    block_shape = (end - start,) if labels else (
                        end - start,
                        values.shape[1],
                    )
                    block_meta = None if labels else sparse.csr_matrix(
                        (0, values.shape[1]), dtype=values.dtype,
                    )
                    delayed_block = delayed(keep_block, pure=False)(future)
                    blocks.append(da.from_delayed(
                        delayed_block,
                        shape=block_shape,
                        dtype=values.dtype,
                        meta=block_meta,
                    ))

                return da.concatenate(blocks, axis=0)

            dask_matrix = distributed_array(matrix)
            dask_labels = distributed_array(labels, labels=True)
            dask_eval_set = None

            if valid_matrix is not None and valid_labels is not None:
                dask_eval_set = [(
                    distributed_array(valid_matrix),
                    distributed_array(valid_labels, labels=True),
                )]

            distributed_model = dxgb.DaskXGBClassifier(**self.params)
            distributed_model.client = self.dask_client

            if dask_eval_set is not None:
                distributed_model.set_params(
                    early_stopping_rounds=int(early_stopping_rounds)
                )
                distributed_model.fit(
                    dask_matrix,
                    dask_labels,
                    eval_set=dask_eval_set,
                    verbose=False,
                )
            else:
                distributed_model.set_params(early_stopping_rounds=None)
                distributed_model.fit(dask_matrix, dask_labels, verbose=False)

            # Persist a normal sklearn-compatible estimator for local inference.
            self.model = xgb.XGBClassifier(**self.params)
            self.model.set_params(device='cpu')
            self.model._Booster = distributed_model.get_booster()
            self.model._Booster.set_param({'device': 'cpu'})
            self.model.n_classes_ = 2
            self.dask_client = None
            del distributed_model, dask_matrix, dask_labels, dask_eval_set
            return self

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
    dask_client=None,
    dask_chunk_rows: int = 65536,
) -> tuple[dict[str, Any], pl.DataFrame]:
    class_counts = np.bincount(labels.astype(np.int8), minlength=2)
    positive_groups = len(np.unique(groups[labels == 1]))
    negative_groups = len(np.unique(groups[labels == 0]))
    splits = min(cv, int(class_counts.min()), positive_groups, negative_groups)

    if splits < 2:
        raise ValueError('Replay-grouped cross-validation requires at least two replay groups per outcome')

    folds = None

    for fold_count in range(splits, 1, -1):
        candidates = list(
            StratifiedGroupKFold(
                n_splits=fold_count,
                shuffle=True,
                random_state=seed,
            ).split(np.zeros(len(labels)), labels, groups)
        )
        valid_folds = all(
            len(np.unique(labels[train_idx])) == 2
            and len(np.unique(labels[valid_idx])) == 2
            and not set(groups[train_idx]).intersection(groups[valid_idx])
            for train_idx, valid_idx in candidates
        )

        if valid_folds:
            splits = fold_count
            folds = candidates
            break

    if folds is None:
        raise ValueError('Could not form replay-grouped CV folds containing both outcomes')

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
            model = AnalyzeRLxG(
                candidate, seed + iteration, jobs, gpu,
                dask_client=dask_client,
                dask_chunk_rows=dask_chunk_rows,
            ).fit(
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
    study = optuna.create_study(
        direction='minimize', sampler=sampler, pruner=optuna.pruners.NopPruner(),
    )

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
    test_frames: pl.DataFrame | None = None,
    dask_client=None,
    dask_chunk_rows: int = 65536,
) -> dict[str, Any]:
    from .evaluate import evaluate_xg_model

    features = prepare_model_data(frames) if not prepared else frames.drop(
        [
            column for column in frames.columns
            if column.startswith(('history_', 'off_', 'pass_', 'passer_'))
            or column.endswith('_scaled')
            or column in {'shooter_team_has_possession', 'shooter_team_possession_length'}
        ],
        strict=False,
    )
    labels = features['is_goal'].to_numpy().astype(np.int8)
    groups = features['replay_id'].to_numpy()

    if features.height < 20 or len(np.unique(labels)) < 2:
        raise ValueError(f'{segment} does not contain enough shot outcomes to train')

    test_features = None if test_frames is None else prepare_model_data(test_frames)

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
    train_matrix = to_sparse_matrix(features[train_idx], feature_columns)
    test_matrix = to_sparse_matrix(
        features[test_idx] if test_features is None else test_features,
        feature_columns,
    )
    output = Path(directory) / segment
    output.mkdir(parents=True, exist_ok=True)
    cv_path = output / f'cross_validation_{segment}.csv'
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
        dask_client,
        dask_chunk_rows,
    )
    model = AnalyzeRLxG(
        best_params, seed, jobs, gpu,
        dask_client=dask_client,
        dask_chunk_rows=dask_chunk_rows,
    ).fit(train_matrix, labels[train_idx])
    test_eval_features = features[test_idx] if test_features is None else test_features
    test_labels = (
        labels[test_idx] if test_features is None
        else test_features['is_goal'].to_numpy().astype(np.int8)
    )
    test_groups = (
        groups[test_idx] if test_features is None
        else test_features['replay_id'].to_numpy()
    )
    shap_matrix = _feature_matrix(test_eval_features, feature_columns)
    column_classes = _feature_column_classes(feature_columns)
    cv_results.write_csv(cv_path)
    evaluation_metrics = evaluate_xg_model(
        model, test_matrix, test_labels, output, segment, feature_columns, shap_matrix,
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
        'cv_strategy': 'stratified_group',
        **evaluation_metrics,
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
    return metrics


@contextmanager
def _dask_client_session(runtime_config: XGRuntimeConfig):
    try:
        from distributed import Client, LocalCluster
    except ImportError as error:
        raise ImportError(
            "Dask training requires `dask[distributed]`; install the project requirements."
        ) from error

    cluster = None
    client = None

    try:
        if runtime_config.dask_scheduler_address:
            client = Client(runtime_config.dask_scheduler_address)

            if runtime_config.dask_workers is not None:
                client.wait_for_workers(runtime_config.dask_workers)
        else:
            gpu = runtime_config.gpu

            if gpu is None:
                try:
                    import torch
                    gpu = torch.cuda.is_available()
                except ImportError:
                    gpu = False

            if gpu:
                try:
                    from dask_cuda import LocalCUDACluster
                except ImportError as error:
                    raise ImportError(
                        "A local multi-GPU Dask cluster requires dask-cuda. "
                        "Install a dask-cuda build compatible with your CUDA/RAPIDS stack, "
                        "or provide dask_scheduler_address for an existing cluster."
                    ) from error

                import torch
                workers = runtime_config.dask_workers or torch.cuda.device_count()

                if workers < 2:
                    raise ValueError(
                        "Dask multi-GPU training needs at least two visible GPUs; "
                        "set dask_scheduler_address to use a remote GPU cluster."
                    )

                cluster = LocalCUDACluster(
                    n_workers=workers,
                    threads_per_worker=1,
                    dashboard_address=None,
                )
            else:
                workers = runtime_config.dask_workers or 2

                if workers < 2:
                    raise ValueError("Dask training requires at least two workers")

                cluster = LocalCluster(
                    n_workers=workers,
                    threads_per_worker=max(1, runtime_config.jobs or 1),
                    dashboard_address=None,
                )

            client = Client(cluster)

        worker_count = len(client.scheduler_info().get('workers', {}))

        if worker_count < 2:
            raise RuntimeError(
                f"Dask xG training requires at least two active workers; found {worker_count}"
            )

        yield client
    finally:
        if client is not None:
            client.close()

        if cluster is not None:
            cluster.close()


def analyzerl_xg(
    frames_folder: str | Path = FRAMES_FOLDER,
    training_parquet: str | Path | None = None,
    directory: str | Path = MODEL_FOLDER,
    model_segments: Sequence[str] | str | None = MODEL_SEGMENTS,
    limit: int | None = None,
    cv: int = 5,
    n_iters: int = 30,
    train_test_split_pct: float = 0.8,
    jobs: int | None = None,
    params: Mapping[str, Any] | None = None,
    random_state: int = RANDOM_STATE,
    gpu: bool | None = None,
    early_stopping_rounds: int = 50,
    train_config: XGTrainConfig | None = None,
    model_config: XGModelConfig | None = None,
    runtime_config: XGRuntimeConfig | None = None,
    test_frames_folder: str | Path | None = None,
    train_replay_ids: Sequence[str] | None = None,
    test_replay_ids: Sequence[str] | None = None,
    train_frame_paths: Sequence[str | Path] | Mapping[str, Sequence[str | Path]] | None = None,
    test_frame_paths: Sequence[str | Path] | Mapping[str, Sequence[str | Path]] | None = None,
    _dask_client=None,
) -> pl.DataFrame:
    # Train and evaluate an xG model for each requested replay segment.
    if runtime_config is not None and runtime_config.dask and _dask_client is None:
        if runtime_config.dask_chunk_rows <= 0:
            raise ValueError('dask_chunk_rows must be a positive integer')

        with _dask_client_session(runtime_config) as client:
            worker_count = len(client.scheduler_info().get('workers', {}))
            print(f"Dask xG training connected to {worker_count} worker(s)", flush=True)
            call_args = locals()
            recursive_args = {
                name: call_args[name]
                for name in analyzerl_xg.__code__.co_varnames[
                    :analyzerl_xg.__code__.co_argcount
                ]
            }
            recursive_args['_dask_client'] = client
            return analyzerl_xg(**recursive_args)

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

    print('Training RL xG model(s)...', flush=True)
    results = []

    for segment in segments:
        # Load, train, and release one mode before scanning data for the next mode.
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
                frame_paths=None if segment_test_paths is None else [Path(path) for path in segment_test_paths],
                selected_replay_ids=test_replay_ids,
            )

        if segment_frames.is_empty():
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
                test_segment_frames,
                dask_client=_dask_client,
                dask_chunk_rows=(
                    runtime_config.dask_chunk_rows
                    if runtime_config is not None else 65536
                ),
            ))
        finally:
            del segment_frames

            if test_segment_frames is not None:
                del test_segment_frames

            gc.collect()

    if not results:
        raise ValueError('No shot events matched the requested model_segments')

    print('Finished RL xG Model(s).', flush=True)
    return pl.DataFrame(results)


if __name__ == '__main__':
    analyzerl_xg(
        n_iters=20,
        model_segments=MODEL_SEGMENTS,
        jobs=2
    )
