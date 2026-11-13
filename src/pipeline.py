"""End-to-end training, evaluation, and multi-learner benchmarking pipelines."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config, VALID_MODEL_TYPES
from src.evaluation import EvaluationReport, UpliftEvaluator
from src.models import BaseMetaLearner, SLearner, TLearner, XLearner
from src.utils import (
    extract_arrays,
    generate_synthetic_uplift_data,
    load_data,
    preprocess_data,
)


@dataclass
class PipelineResult:
    """Output from training and evaluating a single causal meta-learner."""

    model_type: str
    model: BaseMetaLearner
    report: EvaluationReport
    feature_names: List[str]


def _build_learner(model_type: str, config: Config) -> BaseMetaLearner:
    if model_type == "S-learner":
        return SLearner(random_state=config.random_state)
    if model_type == "T-learner":
        return TLearner(random_state=config.random_state)
    if model_type == "X-learner":
        return XLearner(
            propensity_clip_min=config.propensity_clip_min,
            propensity_clip_max=config.propensity_clip_max,
            random_state=config.random_state,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


class UpliftPipeline:
    """Trains and evaluates causal uplift meta-learners on CSV or DataFrame inputs."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    def run_on_dataframe(
        self,
        df: pd.DataFrame,
        model_type: Optional[str] = None,
        true_cate: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """Preprocess DataFrame, split train/test, fit learner, and compute evaluation metrics."""
        chosen_model = model_type or self.config.model_type
        if chosen_model not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Single-model run requires one of {VALID_MODEL_TYPES}, got '{chosen_model}'."
            )

        cleaned = preprocess_data(
            df,
            outcome_var=self.config.outcome_var,
            treatment_var=self.config.treatment_var,
        )
        X, t, y, feature_names = extract_arrays(
            cleaned,
            outcome_var=self.config.outcome_var,
            treatment_var=self.config.treatment_var,
        )

        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.config.test_size,
            stratify=t,
            random_state=self.config.random_state,
        )

        learner = _build_learner(chosen_model, self.config)
        learner.fit(X[train_idx], t[train_idx], y[train_idx])
        uplift_pred = learner.predict(X[test_idx])

        tau_test = true_cate[test_idx] if true_cate is not None else None
        evaluator = UpliftEvaluator(n_deciles=self.config.n_deciles)
        report = evaluator.evaluate(
            y_true=y[test_idx],
            treatment=t[test_idx],
            uplift_pred=uplift_pred,
            true_cate=tau_test,
        )
        return PipelineResult(
            model_type=chosen_model,
            model=learner,
            report=report,
            feature_names=feature_names,
        )

    def run(self) -> PipelineResult:
        """Load data from Config.data_path and run single-model pipeline."""
        if not self.config.data_path:
            raise ValueError("Config.data_path must be set when calling UpliftPipeline.run().")
        raw_df = load_data(self.config.data_path)
        true_cate = raw_df["true_cate"].to_numpy(dtype=float) if "true_cate" in raw_df.columns else None
        return self.run_on_dataframe(raw_df, true_cate=true_cate)


def run_benchmark(
    n_samples: int = 3000,
    config: Optional[Config] = None,
    model_types: Sequence[str] = VALID_MODEL_TYPES,
) -> tuple[pd.DataFrame, Dict[str, PipelineResult]]:
    """Generate synthetic heterogeneous uplift data and benchmark S/T/X-learners."""
    cfg = config or Config()
    df = generate_synthetic_uplift_data(
        n_samples=n_samples,
        random_state=cfg.random_state,
    )
    true_cate = df["true_cate"].to_numpy(dtype=float)
    pipeline = UpliftPipeline(config=cfg)
    results: Dict[str, PipelineResult] = {}
    for m_type in model_types:
        results[m_type] = pipeline.run_on_dataframe(df, model_type=m_type, true_cate=true_cate)
    return df, results
