import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

def load_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data: pd.DataFrame, outcome_var: str, treatment_var: str) -> pd.DataFrame:
    preprocessed_data = data.dropna()
    preprocessed_data = preprocessed_data[(preprocessed_data[outcome_var] != 0) & (preprocessed_data[treatment_var] != 0)]
    return preprocessed_data
