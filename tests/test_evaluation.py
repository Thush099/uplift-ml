"""Tests for UpliftEvaluator metrics (ATE, deciles, Qini, AUUC, PEHE)."""

import json

import numpy as np
import pytest

from src.evaluation import UpliftEvaluator


@pytest.mark.parametrize("n_deciles", [1, 2.5, float("nan"), True])
def test_evaluator_rejects_invalid_decile_count(n_deciles):
    with pytest.raises(ValueError, match="n_deciles must be an integer >= 2"):
        UpliftEvaluator(n_deciles=n_deciles)


def test_evaluator_perfect_vs_reversed_ranking_and_pehe():
    rng = np.random.default_rng(9)
    n = 400
    t = rng.binomial(1, 0.5, size=n)
    true_cate = np.concatenate([np.full(200, 0.6), np.full(200, -0.3)])
    base_prob = 0.4
    prob = np.clip(base_prob + t * true_cate, 0.05, 0.95)
    y = rng.binomial(1, prob).astype(float)

    good_report = UpliftEvaluator(n_deciles=10).evaluate(
        y_true=y,
        treatment=t,
        uplift_pred=true_cate,
        true_cate=true_cate,
    )
    bad_report = UpliftEvaluator(n_deciles=10).evaluate(
        y_true=y,
        treatment=t,
        uplift_pred=-true_cate,
        true_cate=true_cate,
    )

    assert len(good_report.uplift_by_decile) == 10
    assert good_report.pehe == pytest.approx(0.0)
    assert good_report.qini_coefficient > 0.3
    assert good_report.auuc > good_report.random_auuc
    assert good_report.qini_coefficient > bad_report.qini_coefficient
    assert good_report.auuc > bad_report.auuc
    assert bad_report.pehe > 0.5


def test_evaluator_handles_tied_predictions_and_missing_true_cate():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=float)
    t = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    tied_pred = np.full(8, 0.1)

    report = UpliftEvaluator(n_deciles=4).evaluate(y_true=y, treatment=t, uplift_pred=tied_pred)
    assert report.pehe is None
    assert report.true_ate is None
    assert len(report.uplift_by_decile) == 4
    assert report.predicted_ate == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("argument", "bad_values", "expected_message"),
    [
        ("y_true", [np.nan, 0.0], "y_true must contain only finite values."),
        ("uplift_pred", [0.2, np.inf], "uplift_pred must contain only finite values."),
        ("true_cate", [0.1, np.nan], "true_cate must contain only finite values."),
    ],
)
def test_evaluator_rejects_non_finite_values(argument, bad_values, expected_message):
    inputs = {
        "y_true": np.array([1.0, 0.0]),
        "treatment": np.array([1, 0]),
        "uplift_pred": np.array([0.2, -0.1]),
        "true_cate": np.array([0.2, -0.1]),
    }
    inputs[argument] = np.asarray(bad_values)

    with pytest.raises(ValueError, match=expected_message):
        UpliftEvaluator().evaluate(**inputs)


def test_small_imbalanced_groups_report_unavailable_bin_uplift_as_none():
    report = UpliftEvaluator(n_deciles=10).evaluate(
        y_true=np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
        treatment=np.array([1, 0, 1, 0, 1]),
        uplift_pred=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
    )

    assert len(report.uplift_by_decile) == 5
    assert all(decile.n_samples == 1 for decile in report.uplift_by_decile)
    assert all(
        decile.mean_outcome_treated is None
        for decile in report.uplift_by_decile
        if decile.n_treated == 0
    )
    assert all(
        decile.mean_outcome_control is None
        for decile in report.uplift_by_decile
        if decile.n_control == 0
    )
    assert all(decile.observed_uplift is None for decile in report.uplift_by_decile)
    json.dumps(report.to_dict(), allow_nan=False)


def test_evaluator_rejects_missing_arm():
    with pytest.raises(ValueError, match="Both control"):
        UpliftEvaluator().evaluate(
            y_true=np.array([1.0, 0.0]),
            treatment=np.array([1, 1]),
            uplift_pred=np.array([0.2, 0.3]),
        )


def test_pipeline_contracts_and_run(tmp_path) -> None:
    """The learner factory rejects unknown types, run() requires a data path, and a
    real CSV round-trips through the single-model pipeline."""
    from config import Config
    from src.pipeline import UpliftPipeline, _build_learner
    from src.utils import generate_synthetic_uplift_data

    with pytest.raises(ValueError, match="Unsupported model_type"):
        _build_learner("nope", Config())
    with pytest.raises(ValueError, match="data_path must be set"):
        UpliftPipeline(Config()).run()

    df = generate_synthetic_uplift_data(n_samples=120, random_state=2)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    result = UpliftPipeline(Config(data_path=str(csv_path), model_type="T-learner")).run()
    assert result.model_type == "T-learner"
    assert result.feature_names, "features must flow through to the result"
    assert result.report is not None
