"""Data loading, preprocessing, and synthetic heterogeneous uplift dataset generation."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_METADATA_COLS = ("true_cate", "subgroup")


def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load tabular uplift dataset from CSV."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)


def preprocess_data(
    data: pd.DataFrame,
    outcome_var: str = "outcome",
    treatment_var: str = "treatment",
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Validate and clean uplift data while preserving both control (0) and treated (1) groups."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(data).__name__}.")
    if outcome_var not in data.columns:
        raise ValueError(f"Outcome column '{outcome_var}' not found in DataFrame.")
    if treatment_var not in data.columns:
        raise ValueError(f"Treatment column '{treatment_var}' not found in DataFrame.")

    if feature_cols is None:
        excluded = {outcome_var, treatment_var, *_METADATA_COLS}
        selected_features = [c for c in data.columns if c not in excluded]
    else:
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Feature column(s) not found in DataFrame: {missing}")
        selected_features = list(feature_cols)

    if not selected_features:
        raise ValueError("At least one feature column is required for uplift modeling.")

    cols_to_keep = selected_features + [treatment_var, outcome_var]
    cleaned = data[cols_to_keep].dropna().copy()
    if cleaned.empty:
        raise ValueError("No rows remain after dropping missing values.")

    # Normalize boolean treatments cleanly, then enforce binary {0, 1}
    if pd.api.types.is_bool_dtype(cleaned[treatment_var]):
        cleaned[treatment_var] = cleaned[treatment_var].astype(int)
    else:
        cleaned[treatment_var] = pd.to_numeric(cleaned[treatment_var], errors="raise")

    unique_treatments = set(cleaned[treatment_var].unique())
    if not unique_treatments.issubset({0, 1}):
        raise ValueError(
            f"Treatment column '{treatment_var}' must be binary {{0, 1}}, found {sorted(unique_treatments)}."
        )
    if 0 not in unique_treatments or 1 not in unique_treatments:
        raise ValueError(
            "Both control (treatment == 0) and treated (treatment == 1) arms must be present."
        )

    cleaned[treatment_var] = cleaned[treatment_var].astype(int)
    cleaned[outcome_var] = pd.to_numeric(cleaned[outcome_var], errors="raise").astype(float)

    # Encode any categorical / non-numeric feature columns without losing control/treated rows
    non_numeric_features = [
        col for col in selected_features if not pd.api.types.is_numeric_dtype(cleaned[col])
    ]
    if non_numeric_features:
        feature_frame = pd.get_dummies(
            cleaned[selected_features],
            columns=non_numeric_features,
            drop_first=False,
            dtype=float,
        )
    else:
        feature_frame = cleaned[selected_features].astype(float)

    result = feature_frame.copy()
    result[treatment_var] = cleaned[treatment_var].values
    result[outcome_var] = cleaned[outcome_var].values
    return result.reset_index(drop=True)


def extract_arrays(
    data: pd.DataFrame,
    outcome_var: str = "outcome",
    treatment_var: str = "treatment",
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract numeric (X, treatment, outcome, feature_names) arrays from preprocessed data."""
    if feature_cols is None:
        excluded = {outcome_var, treatment_var, *_METADATA_COLS}
        cols = [c for c in data.columns if c not in excluded]
    else:
        cols = list(feature_cols)

    X = data[cols].to_numpy(dtype=float)
    t = data[treatment_var].to_numpy(dtype=int)
    y = data[outcome_var].to_numpy(dtype=float)
    return X, t, y, cols


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate_synthetic_uplift_data(
    n_samples: int = 3000,
    treatment_prob: float = 0.5,
    confounded: bool = False,
    binary_outcome: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic uplift dataset with distinct customer response subgroups.

    Subgroups:
      - persuadables: large positive treatment effect (~ +0.55)
      - sure_things: high baseline conversion, near-zero treatment effect (~ +0.02)
      - sleeping_dogs: negative treatment effect (~ -0.32)
      - lost_causes: low baseline conversion, near-zero treatment effect (~ +0.02)
    """
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20.")
    if not 0.0 < treatment_prob < 1.0:
        raise ValueError("treatment_prob must be strictly between 0 and 1.")

    rng = np.random.default_rng(random_state)

    engagement_score = rng.normal(0.0, 1.0, size=n_samples)
    tenure_months = rng.uniform(1.0, 60.0, size=n_samples)
    recency_days = rng.uniform(1.0, 90.0, size=n_samples)
    past_purchases = rng.poisson(3.0, size=n_samples).astype(float)
    discount_sensitivity = rng.normal(0.0, 1.0, size=n_samples)
    support_tickets = rng.poisson(1.4, size=n_samples).astype(float)

    persuadable_mask = (engagement_score > 0.0) & (discount_sensitivity > -0.15)
    sure_thing_mask = (engagement_score > 0.0) & (~(discount_sensitivity > -0.15))
    sleeping_dog_mask = (engagement_score <= 0.0) & (support_tickets >= 2)
    lost_cause_mask = ~(persuadable_mask | sure_thing_mask | sleeping_dog_mask)

    subgroup = np.empty(n_samples, dtype=object)
    subgroup[persuadable_mask] = "persuadables"
    subgroup[sure_thing_mask] = "sure_things"
    subgroup[sleeping_dog_mask] = "sleeping_dogs"
    subgroup[lost_cause_mask] = "lost_causes"

    mu0 = np.zeros(n_samples, dtype=float)
    mu1 = np.zeros(n_samples, dtype=float)

    # Smooth baseline + heterogeneous response per causal quadrant
    mu0[persuadable_mask] = 0.14 + 0.05 * _sigmoid((tenure_months[persuadable_mask] - 30.0) / 15.0)
    mu1[persuadable_mask] = 0.68 + 0.06 * _sigmoid(discount_sensitivity[persuadable_mask])

    mu0[sure_thing_mask] = 0.80 + 0.05 * _sigmoid(past_purchases[sure_thing_mask] - 3.0)
    mu1[sure_thing_mask] = mu0[sure_thing_mask] + 0.02

    mu0[sleeping_dog_mask] = 0.54 + 0.04 * _sigmoid(engagement_score[sleeping_dog_mask] + 1.0)
    mu1[sleeping_dog_mask] = 0.20 + 0.04 * _sigmoid(engagement_score[sleeping_dog_mask] + 1.0)

    mu0[lost_cause_mask] = 0.09 + 0.03 * _sigmoid((30.0 - recency_days[lost_cause_mask]) / 20.0)
    mu1[lost_cause_mask] = mu0[lost_cause_mask] + 0.02

    mu0 = np.clip(mu0, 0.02, 0.98)
    mu1 = np.clip(mu1, 0.02, 0.98)
    true_cate = mu1 - mu0

    if confounded:
        base_logit = np.log(treatment_prob / (1.0 - treatment_prob))
        propensity = _sigmoid(base_logit + 0.6 * engagement_score - 0.3 * (support_tickets - 1.4))
        treatment = rng.binomial(1, propensity).astype(int)
    else:
        treatment = rng.binomial(1, treatment_prob, size=n_samples).astype(int)

    # Ensure both treatment arms exist even on small sample sizes
    if treatment.sum() == 0:
        treatment[0] = 1
    elif treatment.sum() == n_samples:
        treatment[0] = 0

    expected_outcome = np.where(treatment == 1, mu1, mu0)
    if binary_outcome:
        outcome = rng.binomial(1, expected_outcome).astype(float)
    else:
        outcome = expected_outcome + rng.normal(0.0, 0.1, size=n_samples)

    return pd.DataFrame(
        {
            "engagement_score": np.round(engagement_score, 4),
            "tenure_months": np.round(tenure_months, 2),
            "recency_days": np.round(recency_days, 2),
            "past_purchases": past_purchases,
            "discount_sensitivity": np.round(discount_sensitivity, 4),
            "support_tickets": support_tickets,
            "subgroup": subgroup,
            "treatment": treatment,
            "outcome": outcome,
            "true_cate": np.round(true_cate, 6),
        }
    )
