"""
svm_eval.py — SVM Baseline Evaluation for Feature Selection
Extracted from notebooks/SVM_Classifier.ipynb
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42


def evaluate_svm(X, y, mask=None):
    """
    Train an RBF-SVM on a subset of features defined by mask,
    then evaluate on a stratified 80/20 held-out test set.

    Parameters
    ----------
    X    : np.ndarray  Full feature matrix (samples × features).
    y    : np.ndarray  Labels.
    mask : np.ndarray or None
           Binary mask of length n_features.
           If None, all features are used (baseline SVM).

    Returns
    -------
    dict with keys:
        "accuracy"   : float  Test accuracy.
        "n_selected" : int    Number of features used.
        "n_total"    : int    Total features available.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    n_total = X.shape[1]

    if mask is not None:
        selected_idx = np.where(np.asarray(mask) == 1)[0]
        X_train = X_train[:, selected_idx]
        X_test  = X_test[:, selected_idx]
        n_selected = len(selected_idx)
    else:
        n_selected = n_total

    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_SEED)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return {
        "accuracy":   acc,
        "n_selected": n_selected,
        "n_total":    n_total,
    }
