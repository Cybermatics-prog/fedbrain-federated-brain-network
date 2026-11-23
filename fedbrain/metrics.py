# SPDX-License-Identifier: MIT
"""Subject-level binary evaluation: rows=true HC/ASD, columns=pred HC/ASD."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Non-empty matching one-dimensional labels/probabilities required")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected labels HC=0 ASD=1")
    pred = (probabilities >= .5).astype(int)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) == 2 else None,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "confusion_matrix": cm.tolist(),
        "n_test_subjects": int(len(labels)),
    }


def grouped_metrics(records):
    if not records:
        return {}
    overall = binary_metrics([r['label'] for r in records], [r['asd_probability'] for r in records])
    sites = sorted(set(r['site'] for r in records))
    per_site = {site: binary_metrics([r['label'] for r in records if r['site'] == site],
                                    [r['asd_probability'] for r in records if r['site'] == site])
                for site in sites}
    overall['per_site'] = per_site
    overall['site_mean_accuracy'] = float(np.mean([v['accuracy'] for v in per_site.values()]))
    overall['aggregation'] = 'subject probability = mean window probabilities; macro-site mean reported separately'
    return overall


def evaluator_self_check():
    labels = np.array([0, 1] * 5000)
    gold = binary_metrics(labels, labels.astype(float))
    shuffled = np.random.default_rng(42).permutation(labels)
    control = binary_metrics(labels, shuffled.astype(float))
    assert gold['accuracy'] == 1.0
    assert .47 <= control['accuracy'] <= .53
    return {"gold_accuracy": gold['accuracy'], "shuffled_accuracy": control['accuracy'], "passed": True}
