"""Tests for UpliftEvaluator metrics (ATE, deciles, Qini, AUUC, PEHE)."""

import numpy as np
import pytest

from src.evaluation import UpliftEvaluator


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


def test_evaluator_rejects_missing_arm():
    with pytest.raises(ValueError, match="Both control"):
        UpliftEvaluator().evaluate(
            y_true=np.array([1.0, 0.0]),
            treatment=np.array([1, 1]),
            uplift_pred=np.array([0.2, 0.3]),
        )
