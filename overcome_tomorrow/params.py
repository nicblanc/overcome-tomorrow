import os

MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL_NAME = os.environ.get("MODEL_NAME")
GARMIN_DATA_PREPROC_NAME = os.environ.get("GARMIN_DATA_PREPROC_NAME")
ACTIVITY_PREPROC_NAME = os.environ.get("ACTIVITY_PREPROC_NAME")
DATA_PATH = os.environ.get("DATA_PATH")

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
