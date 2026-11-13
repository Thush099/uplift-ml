"""Public package exports for uplift-ml."""

from config import Config, VALID_MODEL_TYPES
from src.evaluation import DecileMetric, EvaluationReport, UpliftEvaluator
from src.models import BaseMetaLearner, SLearner, TLearner, UpliftModel, XLearner
from src.pipeline import PipelineResult, UpliftPipeline, run_benchmark
from src.utils import (
    extract_arrays,
    generate_synthetic_uplift_data,
    load_data,
    preprocess_data,
)

__all__ = [
    "BaseMetaLearner",
    "Config",
    "DecileMetric",
    "EvaluationReport",
    "PipelineResult",
    "SLearner",
    "TLearner",
    "UpliftEvaluator",
    "UpliftModel",
    "UpliftPipeline",
    "VALID_MODEL_TYPES",
    "XLearner",
    "extract_arrays",
    "generate_synthetic_uplift_data",
    "load_data",
    "preprocess_data",
    "run_benchmark",
]
