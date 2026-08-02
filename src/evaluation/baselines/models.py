"""Baseline predictors for the evaluation harness.

Linear models standardize with a scaler fit only on the fit split. The random
projection baseline is a fixed, seeded projection of the flattened raw window
-- the control that any learned representation must beat.
"""

from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MajorityBaseline:
    """Predicts the majority class of the fit split for every sample."""

    def __init__(self):
        self.cls: Optional[int] = None
        self.p1: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaseline":
        vals, counts = np.unique(np.asarray(y), return_counts=True)
        if len(vals) == 0:
            self.cls = 0
            return self
        self.cls = int(vals[np.argmax(counts)])
        self.p1 = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.cls, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        proba = np.zeros((n, 2), dtype=np.float64)
        proba[:, 1] = self.p1
        proba[:, 0] = 1.0 - self.p1
        return proba


class LogisticBaseline:
    """Logistic regression on a representation, scaler fit on the fit split."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 42):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        self.clf.fit(Xs, np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(np.asarray(X, dtype=np.float64)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(np.asarray(X, dtype=np.float64)))


class RandomProjectionBaseline:
    """Fixed random projection of the flattened raw window -> logistic head.

    A seeded projection is a cheap upper bound on what any linear readout of a
    random-but-large feature space can achieve, and a control every learned
    representation must beat.
    """

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self.proj: Optional[np.ndarray] = None
        self.clf = LogisticBaseline()

    def _project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.proj is None:
            rng = np.random.default_rng(self.seed)
            self.proj = rng.standard_normal((X.shape[1], self.dim)) / np.sqrt(X.shape[1])
        return X @ self.proj

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomProjectionBaseline":
        self.clf.fit(self._project(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self._project(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._project(X))


BASELINE_MODELS = {
    "raw_linear": lambda: LogisticBaseline(),
    "handcrafted_linear": lambda: LogisticBaseline(),
    "random_proj": lambda: RandomProjectionBaseline(),
}
