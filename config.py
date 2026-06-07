from dataclasses import dataclass

@dataclass
class Config:
    data_path: str
    model_type: str
    outcome_var: str
    treatment_var: str

    def __post_init__(self):
        if self.model_type not in ['S-learner', 'T-learner', 'X-learner']:
            raise ValueError('Invalid model type')
