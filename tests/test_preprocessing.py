"""Tests for preprocessing, arm validation, and synthetic uplift DGP."""

import pandas as pd
import pytest

from src.utils import extract_arrays, generate_synthetic_uplift_data, preprocess_data


def test_preprocess_preserves_control_and_zero_outcome_rows():
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "segment": ["A", "B", "A", "B", "A", None],
            "treatment": [0, 0, 1, 1, 0, 1],
            "outcome": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    cleaned = preprocess_data(df, outcome_var="outcome", treatment_var="treatment")

    # Only the last row has NaN; all 5 complete rows (including treatment==0 and outcome==0) survive
    assert len(cleaned) == 5
    assert (cleaned["treatment"] == 0).sum() == 3
    assert (cleaned["treatment"] == 1).sum() == 2
    assert (cleaned["outcome"] == 0.0).sum() == 3
    assert "segment_A" in cleaned.columns and "segment_B" in cleaned.columns

    X, t, y, feature_names = extract_arrays(cleaned)
    assert X.shape == (5, 3)
    assert set(t.tolist()) == {0, 1}
    assert len(y) == 5
    assert len(feature_names) == 3


def test_preprocess_raises_on_missing_arm_or_invalid_treatment():
    df_only_treated = pd.DataFrame(
        {"x1": [1.0, 2.0], "treatment": [1, 1], "outcome": [0.0, 1.0]}
    )
    with pytest.raises(ValueError, match="Both control"):
        preprocess_data(df_only_treated)

    df_only_control = pd.DataFrame(
        {"x1": [1.0, 2.0], "treatment": [0, 0], "outcome": [0.0, 1.0]}
    )
    with pytest.raises(ValueError, match="Both control"):
        preprocess_data(df_only_control)

    df_non_binary = pd.DataFrame(
        {"x1": [1.0, 2.0, 3.0], "treatment": [0, 1, 2], "outcome": [0.0, 1.0, 1.0]}
    )
    with pytest.raises(ValueError, match="binary"):
        preprocess_data(df_non_binary)


def test_synthetic_generator_contains_distinct_causal_subgroups():
    df = generate_synthetic_uplift_data(n_samples=2000, random_state=42)

    subgroups = set(df["subgroup"].unique())
    assert {"persuadables", "sure_things", "sleeping_dogs"}.issubset(subgroups)

    means = df.groupby("subgroup")["true_cate"].mean()
    assert means["persuadables"] > 0.40
    assert abs(means["sure_things"]) < 0.10
    assert means["sleeping_dogs"] < -0.20
    assert set(df["treatment"].unique()) == {0, 1}
