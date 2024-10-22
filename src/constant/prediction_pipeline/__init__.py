import os
from src.constant.s3_bucket import (PREDICTION_BUCKET_NAME,
                                       TRAINING_BUCKET_NAME)
PRED_SCHEMA_FILE_PATH = os.path.join('config', 'prediction_schema.yaml')

PREDICTION_DATA_BUCKET = PREDICTION_BUCKET_NAME
PREDICTION_INPUT_FILE_NAME = "customer_pred_data.csv"
PREDICTION_OUTPUT_FILE_NAME = "customer_predictions.csv"
MODEL_BUCKET_NAME = TRAINING_BUCKET_NAME