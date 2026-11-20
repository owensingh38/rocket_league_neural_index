# Held-out evaluation and reporting for Rocket League expected-goals models.

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def write_evaluation(
    output: Path,
    segment: str,
    model,
    feature_columns: Sequence[str],
    shap_matrix: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
):
    import matplotlib.pyplot as plt

    import shap

    contributions = model.model.get_booster().predict(
        xgb.DMatrix(shap_matrix, feature_names=list(feature_columns)),
        pred_contribs=True,
    )
    shap_values = np.asarray(contributions[:, :-1], dtype=np.float32)
    np.save(output / f'shap_values_{segment}.npy', shap_values)

    shap.summary_plot(
        shap_values,
        shap_matrix,
        feature_names=list(feature_columns),
        max_display=25,
        show=False,
    )
    plt.gcf().set_size_inches(10, 8)
    plt.xlabel('SHAP Value (Impact on Model Output)')
    plt.ylabel('Feature')
    plt.gcf().axes[-1].set_ylabel('Feature Value')
    plt.title(f'AnalyzeRL xG SHAP Summary ({segment.title()})')
    plt.tight_layout()
    plt.savefig(output / f'shap_summary_{segment}.png', dpi=160)
    plt.close()

    dependence_dir = output / 'dependence'
    dependence_dir.mkdir(parents=True, exist_ok=True)
    used_names = set()

    for feature_index, feature in enumerate(feature_columns):
        feature_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(feature)).strip('._')
        feature_name = feature_name or f'feature_{feature_index}'

        if feature_name in used_names:
            feature_name = f'{feature_name}_{feature_index}'

        used_names.add(feature_name)
        shap.dependence_plot(
            feature_index,
            shap_values,
            shap_matrix,
            feature_names=list(feature_columns),
            interaction_index=feature,
            show=False,
            alpha=0.6,
        )
        plt.gcf().set_size_inches(7, 5)
        plt.title(f'{segment.title()} SHAP Dependence: {feature}')
        plt.tight_layout()
        plt.savefig(dependence_dir / f'dependence_{segment}_{feature_name}.png', dpi=120)
        plt.close()

    false_positive, true_positive, _ = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    plt.figure(figsize=(8, 7))
    plt.plot(false_positive, true_positive, label=f'AUC = {auc:.4f}', color='#2f6f73')
    plt.plot([0, 1], [0, 1], '--', color='#777777')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AnalyzeRL xG ROC Curve ({segment.title()})')
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
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Observed Goal Rate')
    plt.title(f'AnalyzeRL xG Calibration Curve ({segment.title()})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f'calibration_{segment}.png', dpi=160)
    plt.close()


def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return the held-out binary classification metrics used by xG reports."""
    return {
        'test_log_loss': float(log_loss(labels, predictions, labels=[0, 1])),
        'test_brier': float(brier_score_loss(labels, predictions)),
        'test_roc_auc': float(roc_auc_score(labels, predictions)),
        'test_average_precision': float(average_precision_score(labels, predictions)),
    }


def evaluate_xg_model(
    model,
    test_matrix: sparse.csr_matrix,
    test_labels: np.ndarray,
    output: Path,
    segment: str,
    feature_columns: Sequence[str],
    shap_matrix: np.ndarray,
) -> dict[str, float]:
    """Generate held-out predictions, metrics, and evaluation plots for xG."""
    predictions = model.predict_proba(test_matrix)[:, 1]
    write_evaluation(
        output, segment, model, feature_columns, shap_matrix, test_labels, predictions,
    )
    return evaluate_predictions(test_labels, predictions)
