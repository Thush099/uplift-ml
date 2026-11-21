"""Causal uplift evaluation metrics: ATE, uplift-by-decile, Qini, AUUC, and PEHE."""

from dataclasses import asdict, dataclass, field
from numbers import Integral
from typing import Any, Dict, List, Optional

import numpy as np

from src.models import VectorLike


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz_fn(y, x))


@dataclass
class DecileMetric:
    """Observed and predicted uplift statistics for one ranked decile bin."""

    decile: int
    n_samples: int
    n_treated: int
    n_control: int
    mean_predicted_uplift: float
    mean_outcome_treated: Optional[float]
    mean_outcome_control: Optional[float]
    observed_uplift: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Comprehensive uplift evaluation results."""

    n_samples: int
    n_treated: int
    n_control: int
    control_mean: float
    treated_mean: float
    observed_ate: float
    predicted_ate: float
    qini_coefficient: float
    qini_auc: float
    random_qini_auc: float
    auuc: float
    random_auuc: float
    auuc_gain: float
    pehe: Optional[float] = None
    true_ate: Optional[float] = None
    uplift_by_decile: List[DecileMetric] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "control_mean": self.control_mean,
            "treated_mean": self.treated_mean,
            "observed_ate": self.observed_ate,
            "predicted_ate": self.predicted_ate,
            "qini_coefficient": self.qini_coefficient,
            "qini_auc": self.qini_auc,
            "random_qini_auc": self.random_qini_auc,
            "auuc": self.auuc,
            "random_auuc": self.random_auuc,
            "auuc_gain": self.auuc_gain,
            "pehe": self.pehe,
            "true_ate": self.true_ate,
            "uplift_by_decile": [d.to_dict() for d in self.uplift_by_decile],
        }


class UpliftEvaluator:
    """Computes ranking, calibration, and heterogeneous effect metrics for uplift predictions."""

    def __init__(self, n_deciles: int = 10) -> None:
        if isinstance(n_deciles, bool) or not isinstance(n_deciles, Integral) or n_deciles < 2:
            raise ValueError("n_deciles must be an integer >= 2.")
        self.n_deciles = int(n_deciles)

    def evaluate(
        self,
        y_true: VectorLike,
        treatment: VectorLike,
        uplift_pred: VectorLike,
        true_cate: Optional[VectorLike] = None,
    ) -> EvaluationReport:
        """Evaluate predicted individual treatment effects against observed outcomes."""
        y = np.asarray(y_true, dtype=float).ravel()
        t = np.asarray(treatment).ravel()
        pred = np.asarray(uplift_pred, dtype=float).ravel()

        if not np.isfinite(y).all():
            raise ValueError("y_true must contain only finite values.")
        if not np.isfinite(pred).all():
            raise ValueError("uplift_pred must contain only finite values.")
        if not (len(y) == len(t) == len(pred)):
            raise ValueError("y_true, treatment, and uplift_pred must have equal length.")
        if len(y) == 0:
            raise ValueError("Cannot evaluate on empty arrays.")

        unique_t = set(np.unique(t))
        if not unique_t.issubset({0, 1}) or 0 not in unique_t or 1 not in unique_t:
            raise ValueError("Both control (0) and treated (1) samples are required for evaluation.")
        t = t.astype(int)

        n_samples = len(y)
        n_treated = int(np.sum(t == 1))
        n_control = int(np.sum(t == 0))
        control_mean = float(np.mean(y[t == 0]))
        treated_mean = float(np.mean(y[t == 1]))
        observed_ate = treated_mean - control_mean
        predicted_ate = float(np.mean(pred))

        qini_auc, random_qini_auc, qini_coeff = self._compute_qini(y, t, pred)
        auuc, random_auuc = self._compute_auuc(y, t, pred)
        deciles = self._compute_uplift_by_decile(y, t, pred)

        pehe: Optional[float] = None
        true_ate: Optional[float] = None
        if true_cate is not None:
            tau = np.asarray(true_cate, dtype=float).ravel()
            if len(tau) != n_samples:
                raise ValueError("true_cate must have the same length as y_true.")
            if not np.isfinite(tau).all():
                raise ValueError("true_cate must contain only finite values.")
            pehe = float(np.sqrt(np.mean((pred - tau) ** 2)))
            true_ate = float(np.mean(tau))

        return EvaluationReport(
            n_samples=n_samples,
            n_treated=n_treated,
            n_control=n_control,
            control_mean=control_mean,
            treated_mean=treated_mean,
            observed_ate=observed_ate,
            predicted_ate=predicted_ate,
            qini_coefficient=qini_coeff,
            qini_auc=qini_auc,
            random_qini_auc=random_qini_auc,
            auuc=auuc,
            random_auuc=random_auuc,
            auuc_gain=auuc - random_auuc,
            pehe=pehe,
            true_ate=true_ate,
            uplift_by_decile=deciles,
        )

    @staticmethod
    def _qini_curve_values(y_sorted: np.ndarray, t_sorted: np.ndarray) -> np.ndarray:
        n = len(y_sorted)
        n_t = np.cumsum(t_sorted)
        n_c = np.cumsum(1 - t_sorted)
        y_t = np.cumsum(y_sorted * t_sorted)
        y_c = np.cumsum(y_sorted * (1 - t_sorted))

        q = np.zeros(n + 1, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled_control = np.where(n_c > 0, y_c * (n_t / np.maximum(n_c, 1)), 0.0)
        q[1:] = (y_t - scaled_control) / n
        return q

    def _compute_qini(
        self, y: np.ndarray, t: np.ndarray, pred: np.ndarray
    ) -> tuple[float, float, float]:
        n = len(y)
        x = np.arange(0, n + 1, dtype=float) / n

        order_model = np.argsort(-pred, kind="mergesort")
        q_model = self._qini_curve_values(y[order_model], t[order_model])
        qini_auc = _trapezoid(q_model, x)
        random_qini_auc = 0.5 * float(q_model[-1])

        # Oracle upper bound Qini ordering: treated high-outcome first, then low-outcome controls
        oracle_score = np.where(t == 1, y, -y)
        order_opt = np.lexsort((t, -oracle_score))
        q_opt = self._qini_curve_values(y[order_opt], t[order_opt])
        opt_qini_auc = _trapezoid(q_opt, x)

        denom = opt_qini_auc - random_qini_auc
        qini_coeff = (qini_auc - random_qini_auc) / denom if abs(denom) > 1e-12 else 0.0
        return qini_auc, random_qini_auc, float(qini_coeff)

    @staticmethod
    def _compute_auuc(y: np.ndarray, t: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        n = len(y)
        x = np.arange(0, n + 1, dtype=float) / n
        order = np.argsort(-pred, kind="mergesort")
        y_s = y[order]
        t_s = t[order]

        n_t = np.cumsum(t_s)
        n_c = np.cumsum(1 - t_s)
        y_t = np.cumsum(y_s * t_s)
        y_c = np.cumsum(y_s * (1 - t_s))

        u = np.zeros(n + 1, dtype=float)
        valid = (n_t > 0) & (n_c > 0)
        k = np.arange(1, n + 1, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_diff = np.where(
                valid,
                (y_t / np.maximum(n_t, 1)) - (y_c / np.maximum(n_c, 1)),
                0.0,
            )
        u[1:] = rate_diff * (k / n)

        auuc = _trapezoid(u, x)
        random_auuc = 0.5 * float(u[-1])
        return auuc, random_auuc

    def _compute_uplift_by_decile(
        self, y: np.ndarray, t: np.ndarray, pred: np.ndarray
    ) -> List[DecileMetric]:
        order = np.argsort(-pred, kind="mergesort")
        n_bins = min(self.n_deciles, len(order))
        bins = np.array_split(order, n_bins)

        deciles: List[DecileMetric] = []
        for idx, bin_indices in enumerate(bins, start=1):
            t_bin = t[bin_indices]
            y_bin = y[bin_indices]
            p_bin = pred[bin_indices]
            t_mask = t_bin == 1
            c_mask = t_bin == 0
            n_t = int(np.sum(t_mask))
            n_c = int(np.sum(c_mask))
            y_t_mean = float(np.mean(y_bin[t_mask])) if n_t > 0 else None
            y_c_mean = float(np.mean(y_bin[c_mask])) if n_c > 0 else None
            obs_uplift = (
                y_t_mean - y_c_mean
                if y_t_mean is not None and y_c_mean is not None
                else None
            )
            deciles.append(
                DecileMetric(
                    decile=idx,
                    n_samples=int(len(bin_indices)),
                    n_treated=n_t,
                    n_control=n_c,
                    mean_predicted_uplift=float(np.mean(p_bin)),
                    mean_outcome_treated=y_t_mean,
                    mean_outcome_control=y_c_mean,
                    observed_uplift=obs_uplift,
                )
            )
        return deciles
