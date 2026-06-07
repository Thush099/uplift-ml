import argparse
import logging
from src.pipeline import UpliftPipeline
from config import Config

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Uplift ML Framework')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--model_type', type=str, help='Type of model to use (S/T/X-learner)')
    parser.add_argument('--outcome_var', type=str, help='Name of outcome variable')
    parser.add_argument('--treatment_var', type=str, help='Name of treatment variable')
    args = parser.parse_args()

    config = Config(data_path=args.data_path, model_type=args.model_type, outcome_var=args.outcome_var, treatment_var=args.treatment_var)
    pipeline = UpliftPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()
