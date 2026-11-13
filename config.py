"""Configuration dataclass for causal uplift modeling pipelines."""

from dataclasses import dataclass
from typing import Optional, Tuple

VALID_MODEL_TYPES: Tuple[str, ...] = ("S-learner", "T-learner", "X-learner")


@dataclass
class Config:
    """Configuration for uplift training, propensity clipping, and evaluation."""

    data_path: Optional[str] = None
    model_type: str = "X-learner"
    outcome_var: str = "outcome"
    treatment_var: str = "treatment"
    test_size: float = 0.3
    random_state: int = 42
    n_deciles: int = 10
    propensity_clip_min: float = 0.01
    propensity_clip_max: float = 0.99

    def __post_init__(self) -> None:
        if self.model_type not in VALID_MODEL_TYPES and self.model_type != "all":
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. Expected one of {VALID_MODEL_TYPES} or 'all'."
            )
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be strictly between 0 and 1.")
        if not 0.0 < self.propensity_clip_min < self.propensity_clip_max < 1.0:
            raise ValueError("Propensity clipping bounds must satisfy 0 < min < max < 1.")
        if self.n_deciles < 2:
            raise ValueError("n_deciles must be at least 2.")
