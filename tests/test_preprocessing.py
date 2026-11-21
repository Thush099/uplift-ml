"""Tests for preprocessing, arm validation, and synthetic uplift DGP."""

import math

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


@pytest.mark.parametrize(
    ("feature_cols", "message"),
    [
        (["outcome", "x1"], "cannot be used as feature columns"),
        (["treatment", "x1"], "cannot be used as feature columns"),
        (["x1", "x1"], "must not contain duplicates"),
    ],
)
def test_preprocess_rejects_invalid_feature_columns(feature_cols, message):
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "treatment": [0, 1, 0, 1],
            "outcome": [0.0, 1.0, 1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match=message):
        preprocess_data(df, feature_cols=feature_cols)


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


def test_preprocess_rejects_non_finite_outcome_and_treatment_with_locations():
    df_nan_outcome = pd.DataFrame(
        {"x1": [1.0, 2.0, 3.0], "treatment": [0, 1, 0], "outcome": [0.0, float("nan"), 1.0]}
    )
    with pytest.raises(ValueError, match=r"Column 'outcome' contains non-finite value\(s\) at row position\(s\) \[1\]"):
        preprocess_data(df_nan_outcome)

    df_inf_outcome = pd.DataFrame(
        {"x1": [1.0, 2.0], "treatment": [0, 1], "outcome": [float("inf"), 1.0]}
    )
    with pytest.raises(ValueError, match=r"Column 'outcome' contains non-finite value\(s\) at row position\(s\) \[0\]"):
        preprocess_data(df_inf_outcome)

    df_nan_treatment = pd.DataFrame(
        {"x1": [1.0, 2.0], "treatment": [float("nan"), 1], "outcome": [0.0, 1.0]}
    )
    with pytest.raises(ValueError, match=r"Column 'treatment' contains non-finite value"):
        preprocess_data(df_nan_treatment)


def test_preprocess_rejects_infinite_feature_values_with_locations():
    df_inf_feature = pd.DataFrame(
        {"x1": [1.0, float("-inf"), 3.0], "treatment": [0, 1, 0], "outcome": [0.0, 1.0, 1.0]}
    )
    with pytest.raises(ValueError, match=r"Feature column 'x1' contains infinite value\(s\) at row position\(s\) \[1\]"):
        preprocess_data(df_inf_feature)


def test_preprocess_still_drops_missing_feature_rows_silently():
    df = pd.DataFrame(
        {
            "x1": [1.0, float("nan"), 3.0, 4.0],
            "treatment": [0, 1, 0, 1],
            "outcome": [0.0, 1.0, 1.0, 1.0],
        }
    )
    cleaned = preprocess_data(df)
    assert len(cleaned) == 3


def test_load_data_reports_missing_file_by_name(tmp_path):
    from src.utils import load_data

    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        load_data(missing)


def test_load_data_reads_csv(tmp_path):
    from src.utils import load_data

    path = tmp_path / "data.csv"
    path.write_text("outcome,treatment,x1\n1,0,0.5\n0,1,1.5\n")
    df = load_data(path)
    assert list(df.columns) == ["outcome", "treatment", "x1"]
    assert len(df) == 2


def test_run_benchmark_returns_one_result_per_requested_model():
    from config import Config
    from src.pipeline import run_benchmark

    df, results = run_benchmark(n_samples=200, config=Config(), model_types=["S-learner"])
    assert list(results) == ["S-learner"]
    assert len(df) == 200
    assert "qini" in df.columns or "true_cate" in df.columns
    result = results["S-learner"]
    assert math.isfinite(result.report.qini_coefficient)
    assert result.report.n_samples > 0
