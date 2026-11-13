"""Causal uplift meta-learner implementations: S-Learner, T-Learner, and X-Learner."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ArrayLike = Union[np.ndarray, pd.DataFrame]
VectorLike = Union[np.ndarray, pd.Series]


def _default_outcome_estimator(random_state: int = 42) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        random_state=random_state,
    )


def _default_effect_estimator(random_state: int = 42) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=15,
        random_state=random_state,
    )


def _validate_inputs(
    X: ArrayLike,
    treatment: VectorLike,
    outcome: VectorLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D feature matrix.")

    t_arr = np.asarray(treatment).ravel()
    y_arr = np.asarray(outcome, dtype=float).ravel()

    if len(t_arr) != X_arr.shape[0] or len(y_arr) != X_arr.shape[0]:
        raise ValueError("X, treatment, and outcome must have identical row counts.")

    unique_t = set(np.unique(t_arr))
    if not unique_t.issubset({0, 1}):
        raise ValueError(f"treatment must contain only binary values {{0, 1}}, got {sorted(unique_t)}.")

    t_int = t_arr.astype(int)
    if np.sum(t_int == 0) == 0 or np.sum(t_int == 1) == 0:
        raise ValueError(
            "Both control (treatment == 0) and treated (treatment == 1) groups must contain at least one sample."
        )

    return X_arr, t_int, y_arr


class BaseMetaLearner(ABC):
    """Abstract base class for causal uplift meta-learners."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(self, X: ArrayLike, treatment: VectorLike, outcome: VectorLike) -> "BaseMetaLearner":
        """Fit meta-learner on covariates, binary treatment assignment, and outcome."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Estimate individual Conditional Average Treatment Effects (CATE)."""

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before calling predict.")


class SLearner(BaseMetaLearner):
    """Single-model meta-learner trained on [X, T]; predicts E[Y|X, T=1] - E[Y|X, T=0]."""

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.base_estimator = base_estimator
        self.estimator_: Optional[BaseEstimator] = None

    def fit(self, X: ArrayLike, treatment: VectorLike, outcome: VectorLike) -> "SLearner":
        X_arr, t_arr, y_arr = _validate_inputs(X, treatment, outcome)
        template = (
            clone(self.base_estimator)
            if self.base_estimator is not None
            else _default_outcome_estimator(self.random_state)
        )
        X_aug = np.column_stack([X_arr, t_arr])
        template.fit(X_aug, y_arr)
        self.estimator_ = template
        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        assert self.estimator_ is not None
        X_arr = np.asarray(X, dtype=float)
        n = X_arr.shape[0]
        X_treated = np.column_stack([X_arr, np.ones(n, dtype=float)])
        X_control = np.column_stack([X_arr, np.zeros(n, dtype=float)])
        return self.estimator_.predict(X_treated) - self.estimator_.predict(X_control)


class TLearner(BaseMetaLearner):
    """Two-model meta-learner fitting separate mu_0 (T=0) and mu_1 (T=1) estimators."""

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.base_estimator = base_estimator
        self.mu_0_: Optional[BaseEstimator] = None
        self.mu_1_: Optional[BaseEstimator] = None

    def fit(self, X: ArrayLike, treatment: VectorLike, outcome: VectorLike) -> "TLearner":
        X_arr, t_arr, y_arr = _validate_inputs(X, treatment, outcome)
        template = self.base_estimator or _default_outcome_estimator(self.random_state)
        self.mu_0_ = clone(template)
        self.mu_1_ = clone(template)

        control_mask = t_arr == 0
        treated_mask = t_arr == 1
        self.mu_0_.fit(X_arr[control_mask], y_arr[control_mask])
        self.mu_1_.fit(X_arr[treated_mask], y_arr[treated_mask])
        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        assert self.mu_0_ is not None and self.mu_1_ is not None
        X_arr = np.asarray(X, dtype=float)
        return self.mu_1_.predict(X_arr) - self.mu_0_.predict(X_arr)


class XLearner(BaseMetaLearner):
    """Three-stage X-Learner (Kunzel et al., 2019) with propensity weighting.

    Stage 1: Fit outcome models mu_0 on control (T=0) and mu_1 on treated (T=1).
    Stage 2: Compute imputed effects D_1 = Y_1 - mu_0(X_1) and D_0 = mu_1(X_0) - Y_0;
             fit tau_1 on (X_1, D_1) and tau_0 on (X_0, D_0).
    Stage 3: Blend g(X)*tau_0(X) + (1 - g(X))*tau_1(X) with clipped propensity g(X).
    """

    def __init__(
        self,
        outcome_estimator: Optional[BaseEstimator] = None,
        effect_estimator: Optional[BaseEstimator] = None,
        propensity_clip_min: float = 0.01,
        propensity_clip_max: float = 0.99,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        if not 0.0 < propensity_clip_min < propensity_clip_max < 1.0:
            raise ValueError("propensity clipping bounds must satisfy 0 < min < max < 1.")
        self.outcome_estimator = outcome_estimator
        self.effect_estimator = effect_estimator
        self.propensity_clip_min = propensity_clip_min
        self.propensity_clip_max = propensity_clip_max

        self.mu_0_: Optional[BaseEstimator] = None
        self.mu_1_: Optional[BaseEstimator] = None
        self.tau_0_: Optional[BaseEstimator] = None
        self.tau_1_: Optional[BaseEstimator] = None
        self.propensity_model_: Optional[Pipeline] = None

    def fit(self, X: ArrayLike, treatment: VectorLike, outcome: VectorLike) -> "XLearner":
        X_arr, t_arr, y_arr = _validate_inputs(X, treatment, outcome)
        control_mask = t_arr == 0
        treated_mask = t_arr == 1

        X_0, y_0 = X_arr[control_mask], y_arr[control_mask]
        X_1, y_1 = X_arr[treated_mask], y_arr[treated_mask]

        # Stage 1: response surface estimation
        outcome_tpl = self.outcome_estimator or _default_outcome_estimator(self.random_state)
        self.mu_0_ = clone(outcome_tpl)
        self.mu_1_ = clone(outcome_tpl)
        self.mu_0_.fit(X_0, y_0)
        self.mu_1_.fit(X_1, y_1)

        # Stage 2: imputed individual treatment effects and CATE regressions
        imputed_1 = y_1 - self.mu_0_.predict(X_1)
        imputed_0 = self.mu_1_.predict(X_0) - y_0

        effect_tpl = self.effect_estimator or _default_effect_estimator(self.random_state)
        self.tau_0_ = clone(effect_tpl)
        self.tau_1_ = clone(effect_tpl)
        self.tau_0_.fit(X_0, imputed_0)
        self.tau_1_.fit(X_1, imputed_1)

        # Stage 3: propensity model g(X) = P(T=1 | X)
        self.propensity_model_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(max_iter=1000, random_state=self.random_state),
                ),
            ]
        )
        self.propensity_model_.fit(X_arr, t_arr)
        self.is_fitted_ = True
        return self

    def predict_propensity(self, X: ArrayLike) -> np.ndarray:
        """Return clipped treatment assignment probability g(X) in [clip_min, clip_max]."""
        self._check_is_fitted()
        assert self.propensity_model_ is not None
        X_arr = np.asarray(X, dtype=float)
        raw_prob = self.propensity_model_.predict_proba(X_arr)[:, 1]
        return np.clip(raw_prob, self.propensity_clip_min, self.propensity_clip_max)

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        assert self.tau_0_ is not None and self.tau_1_ is not None
        X_arr = np.asarray(X, dtype=float)
        g_hat = self.predict_propensity(X_arr)
        return g_hat * self.tau_0_.predict(X_arr) + (1.0 - g_hat) * self.tau_1_.predict(X_arr)


class UpliftModel:
    """Factory wrapper supporting 'S-learner', 'T-learner', and 'X-learner' identifiers."""

    _LEARNERS = {
        "S-learner": SLearner,
        "T-learner": TLearner,
        "X-learner": XLearner,
    }

    def __init__(self, model_type: str, random_state: int = 42) -> None:
        if model_type not in self._LEARNERS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose from {tuple(self._LEARNERS.keys())}."
            )
        self.model_type = model_type
        self.learner: BaseMetaLearner = self._LEARNERS[model_type](random_state=random_state)

    def fit(self, X: ArrayLike, treatment: VectorLike, outcome: VectorLike) -> "UpliftModel":
        self.learner.fit(X, treatment, outcome)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.learner.predict(X)
