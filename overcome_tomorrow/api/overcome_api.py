from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from overcome_tomorrow.params import *
from overcome_tomorrow.ml_logic.model import *
from overcome_tomorrow.ml_logic.preprocess import *
from datetime import datetime, timedelta
import pandas as pd


tomorrow_app = FastAPI()
tomorrow_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

tags_metadata = [
    {
        "name": "models",
        "description": "Operations regarding overcome-tomorrow models.",
    },
    {
        "name": "activities",
        "description": "Operations regarding activities.",
    }
]

DEFAULT_ACTIVITY = {
    "timestamp": "08/10/2023 17:16",
    "total_anaerobic_training_effect": 0.1,
    "enhanced_avg_speed": 4.351,
    "avg_stroke_distance": None,
    "total_training_effect": 5.0,
    "num_active_lengths": None,
    "max_running_cadence": 95.0,
    "max_heart_rate": 174.0,
    "sub_sport": "generic",
    "total_descent": 78.0,
    "start_time": "08/10/2023 14:31",
    "total_distance": 42685.7,
    "pool_length": None,
    178: 2724.0,
    "total_calories": 2512.0,
    "max_cadence": 95.0,
    "sport": "running",
    188: 3.0,
    "avg_step_length": 1476.1,
    "enhanced_max_speed": 7.81,
    205: 100.0,
    "num_lengths": None,
    206: 40.0,
    207: 40.0,
    "pool_length_unit": None,
    "normalized_power": 385.0,
    "training_load_peak": round(309.48114013671875, 2),
    "total_ascent": 85.0,
    "avg_power": 379.0,
    "total_strokes": None,
    "training_stress_score": None,
    "avg_cadence": 88.0,
    "avg_heart_rate": 153.0,
    "max_power": 825.0,
    "activity_id": "nicko64@hotmail.fr_213811102683"
}

preproc_garmin_data, preproc_activity, model = load_preprocessors_and_model(
    MODEL_PATH)
garmin_data, activities = get_data(DATA_PATH)


@tomorrow_app.get("/models", tags=["models"])
def get_model_names():
    return ["model1", "model2", "model3"]


@tomorrow_app.get("/activities", tags=["activities"])
def get_activity(activity_datetime: str, summarized: bool = True):
    # TODO handle date
    # TODO return multiple activities.
    if summarized:
        return [DEFAULT_ACTIVITY]
    activity_df = pd.read_csv(
        "overcome_tomorrow/api/static_data/nicko64@hotmail.fr_213811102683.csv", parse_dates=True, index_col="timestamp")
    return [activity_df.to_json()]


@tomorrow_app.get("/activities/next", tags=["activities"])
def predict_next_activity_for_models(models_name: str = "DEFAULT"):
    # TODO get activity for each model
    return [predict_next_activity(model_name) for model_name in models_name.split(",")]


@tomorrow_app.get("/activities/next/{model_name}", tags=["activities"])
def predict_next_activity(model_name: str):
    # TODO get activity for given model
    # TODO handle 'DEFAULT' model
    return predict_for_last_n_days(garmin_data, preproc_garmin_data, preproc_activity, model, 1).iloc[0].to_json()


@tomorrow_app.get("/activities/date", tags=["activities"])
def predict_activity_for_date(date: datetime = datetime.now()):
    return predict_for_date(garmin_data, preproc_garmin_data, preproc_activity, model, date).iloc[0].to_json()


@tomorrow_app.get("/")
def root():
    return {"Overcome": "Tomorrow"}
