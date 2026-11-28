"""Tests for SLearner, TLearner, and XLearner meta-learners."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.evaluation import UpliftEvaluator
from src.models import SLearner, TLearner, UpliftModel, XLearner
from src.utils import extract_arrays, generate_synthetic_uplift_data, preprocess_data


@pytest.mark.parametrize("learner_cls", [SLearner, TLearner, XLearner])
def test_learners_achieve_positive_qini_auuc_and_accurate_ate(learner_cls):
    df = generate_synthetic_uplift_data(n_samples=2500, random_state=42)
    true_cate = df["true_cate"].to_numpy(dtype=float)
    cleaned = preprocess_data(df)
    X, t, y, _ = extract_arrays(cleaned)

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, stratify=t, random_state=42)

    model = learner_cls(random_state=42)
    model.fit(X[train_idx], t[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])

    report = UpliftEvaluator().evaluate(
        y_true=y[test_idx],
        treatment=t[test_idx],
        uplift_pred=preds,
        true_cate=true_cate[test_idx],
    )

    oracle_report = UpliftEvaluator().evaluate(
        y_true=y[test_idx],
        treatment=t[test_idx],
        uplift_pred=true_cate[test_idx],
        true_cate=true_cate[test_idx],
    )

    assert report.qini_coefficient > 0.15
    assert report.qini_coefficient >= 0.70 * oracle_report.qini_coefficient
    assert report.auuc > 0.0
    assert report.auuc_gain > 0.0
    assert abs(report.predicted_ate - report.true_ate) < 0.06
    assert report.pehe is not None and report.pehe < 0.25
    assert report.uplift_by_decile[0].observed_uplift > report.uplift_by_decile[-1].observed_uplift


def test_xlearner_handles_imbalanced_treatment_split_and_clips_propensity():
    df = generate_synthetic_uplift_data(
        n_samples=2000,
        treatment_prob=0.15,
        confounded=True,
        random_state=123,
    )
    true_cate = df["true_cate"].to_numpy(dtype=float)
    cleaned = preprocess_data(df)
    X, t, y, _ = extract_arrays(cleaned)

    assert 0.05 < t.mean() < 0.30

    x_learner = XLearner(
        propensity_clip_min=0.01,
        propensity_clip_max=0.99,
        random_state=123,
    )
    x_learner.fit(X, t, y)
    propensities = x_learner.predict_propensity(X)
    preds = x_learner.predict(X)

    assert propensities.min() >= 0.01
    assert propensities.max() <= 0.99
    report = UpliftEvaluator().evaluate(y_true=y, treatment=t, uplift_pred=preds, true_cate=true_cate)
    assert report.qini_coefficient > 0.20
    assert report.auuc > 0.0


@pytest.mark.parametrize("learner_cls", [SLearner, TLearner, XLearner])
def test_missing_treatment_arm_raises_value_error(learner_cls):
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.random.default_rng(0).integers(0, 2, size=20).astype(float)

    with pytest.raises(ValueError, match="Both control"):
        learner_cls().fit(X, np.zeros(20, dtype=int), y)

    with pytest.raises(ValueError, match="Both control"):
        learner_cls().fit(X, np.ones(20, dtype=int), y)


def test_predict_before_fit_and_factory_validation():
    for learner in (SLearner(), TLearner(), XLearner()):
        with pytest.raises(RuntimeError, match="fitted"):
            learner.predict(np.zeros((5, 3)))

    with pytest.raises(ValueError, match="Unknown model_type"):
        UpliftModel("Invalid-learner")


def test_uplift_model_wrapper_contract() -> None:
    """UpliftModel rejects unknown learners and passes fit/predict through."""
    import numpy as np

    from src.models import UpliftModel

    with pytest.raises(ValueError, match="Unknown model_type"):
        UpliftModel(model_type="nope")

    X = np.array([[0.1], [0.2], [0.3], [0.4]] * 5)
    treatment = np.array([0, 1] * 10)
    outcome = np.array([0.0, 1.0] * 10)
    model = UpliftModel(model_type="T-learner", random_state=0)
    assert model.fit(X, treatment, outcome) is model
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()
