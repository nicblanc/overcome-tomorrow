import os

##################  ENV VARIABLES  ##################

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

DTYPES_GARMIN_DATA_RAW = {
    "start_sleep": "datetime64[ns, UTC]",
    "end_sleep": "datetime64[ns, UTC]",
    "beginTimestamp": "datetime64[ns, UTC]"
}

DTYPES_ACTIVITIES_RAW = {
    "timestamp": "datetime64[ns, UTC]",
    "start_time": "datetime64[ns, UTC]"
}


##################  CONSTANTS  #####################


MODEL_FILENAME_KEY = "model_filename"
PREPROC_GARMIN_DATA_FILENAME_KEY = "preproc_garmin_data_filename"
PREPROC_ACTIVITY_FILENAME_KEY = "preproc_activity_filename"

MODEL_KEY = "model"
PREPROC_GARMIN_DATA_KEY = "preproc_garmin_data"
PREPROC_ACTIVITY_KEY = "preproc_activity"

MODEL_BLOB_UPDATED_KEY = "model_blob_updated"
PREPROC_GARMIN_DATA_BLOB_UPDATED_KEY = "preproc_garmin_data_blob_updated"
PREPROC_ACTIVITY_BLOB_UPDATED_KEY = "preproc_activity_blob_updated"
