import logging
from src.models import UpliftModel
from src.utils import load_data, preprocess_data
from config import Config

logging.basicConfig(level=logging.INFO)

class UpliftPipeline:
    def __init__(self, config: Config):
        self.config = config

    def run(self):
        data = load_data(self.config.data_path)
        preprocessed_data = preprocess_data(data, self.config.outcome_var, self.config.treatment_var)
        model = UpliftModel(self.config.model_type)
        model.train(preprocessed_data)
        treatment_effect = model.evaluate(preprocessed_data)
        logging.info(f'Treatment effect estimate: {treatment_effect}')
