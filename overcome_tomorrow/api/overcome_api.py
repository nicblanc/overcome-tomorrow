from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from overcome_tomorrow.params import *
from overcome_tomorrow.ml_logic.model import *
from datetime import datetime, timedelta
import pandas as pd
import schedule


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

models_dict = get_all_models()
garmin_data, activities = get_data()


def check_models_updated():
    print("⌛ Checking if models and preprocessors are up to date")
    global models_dict
    for model_dict in models_dict.values():
        model_blob_updated_tmp, preproc_garmin_data_blob_updated_tmp, preproc_activity_blob_updated_tmp = get_last_modified_dates_for_model_and_preprocessors_from_gcs(
            model_filename=model_dict[MODEL_FILENAME_KEY],
            preproc_garmin_data_filename=model_dict[PREPROC_GARMIN_DATA_FILENAME_KEY],
            preproc_activity_filename=model_dict[PREPROC_ACTIVITY_FILENAME_KEY]
        )
        if (model_blob_updated_tmp > model_dict[MODEL_BLOB_UPDATED_KEY]) or \
            (preproc_garmin_data_blob_updated_tmp > model_dict[PREPROC_GARMIN_DATA_BLOB_UPDATED_KEY]) or \
                (preproc_activity_blob_updated_tmp > model_dict[PREPROC_ACTIVITY_BLOB_UPDATED_KEY]):
            download_model_and_preprocessors_from_gcs(
                model_filename=model_dict[MODEL_FILENAME_KEY],
                preproc_garmin_data_filename=model_dict[PREPROC_GARMIN_DATA_FILENAME_KEY],
                preproc_activity_filename=model_dict[PREPROC_ACTIVITY_FILENAME_KEY]
            )
            preproc_garmin_data, preproc_activity, model = load_preprocessors_and_model(
                model_filename=model_dict[MODEL_FILENAME_KEY],
                preproc_garmin_data_filename=model_dict[PREPROC_GARMIN_DATA_FILENAME_KEY],
                preproc_activity_filename=model_dict[PREPROC_ACTIVITY_FILENAME_KEY]
            )
            model_dict[MODEL_KEY] = model
            model_dict[PREPROC_GARMIN_DATA_KEY] = preproc_garmin_data
            model_dict[PREPROC_ACTIVITY_KEY] = preproc_activity

            model_dict[MODEL_BLOB_UPDATED_KEY] = model_blob_updated_tmp
            model_dict[PREPROC_GARMIN_DATA_BLOB_UPDATED_KEY] = preproc_garmin_data_blob_updated_tmp
            model_dict[PREPROC_ACTIVITY_BLOB_UPDATED_KEY] = preproc_activity_blob_updated_tmp


schedule.every(1).minutes.do(check_models_updated)


@tomorrow_app.get("/models", tags=["models"])
def get_model_names():
    return set(models_dict.keys())


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
def predict_next_activity_for_models(models_name: str):
    # TODO get activity for each model
    models_name = models_name.strip()
    if len(models_name) > 0:
        return [predict_next_activity(model_name) for model_name in models_name.split(",")]
    return [predict_next_activity(model_name) for model_name in models_dict.keys()]


@tomorrow_app.get("/activities/next/compare", tags=["activities"])
def predict_next_activity_for_models(models_name: str):
    # TODO get activity for each model
    models_name = models_name.strip()
    if len(models_name) > 0:
        return [compare_next_activity(model_name) for model_name in models_name.split(",")]
    return [compare_next_activity(model_name) for model_name in models_dict.keys()]


@tomorrow_app.get("/activities/next/{model_name}", tags=["activities"])
def predict_next_activity(model_name: str):
    schedule.run_pending()
    # TODO handle 'DEFAULT' or None model
    preproc_garmin_data, preproc_activity, model = get_model_from_dict(
        model_name)

    return predict_for_last_n_days(garmin_data, preproc_garmin_data, preproc_activity, model, 1).iloc[0].to_json()


@tomorrow_app.get("/activities/next/{model_name}/compare", tags=["activities"])
def compare_next_activity(model_name: str):
    schedule.run_pending()
    # TODO handle 'DEFAULT' or None model
    preproc_garmin_data, preproc_activity, model = get_model_from_dict(
        model_name)

    return predict_vs_real_for_last_n_days(garmin_data, activities, preproc_garmin_data, preproc_activity, model, 1).to_json()


@tomorrow_app.get("/activities/date", tags=["activities"])
def predict_activity_for_date(model_name: str, date: datetime = datetime.now()):
    schedule.run_pending()
    # TODO handle 'DEFAULT' or None model
    preproc_garmin_data, preproc_activity, model = get_model_from_dict(
        model_name)
    return predict_for_date(garmin_data, preproc_garmin_data, preproc_activity, model, date).iloc[0].to_json()


@tomorrow_app.get("/activities/date/compare", tags=["activities"])
def compare_activity_for_date(model_name: str, date: datetime = datetime.now()):
    schedule.run_pending()
    # TODO handle 'DEFAULT' or None model
    preproc_garmin_data, preproc_activity, model = get_model_from_dict(
        model_name)
    return predict_vs_real_for_date(garmin_data, activities, preproc_garmin_data, preproc_activity, model, date).to_json()


def get_model_from_dict(model_name):
    global models_dict
    model_name = model_name.strip()
    if model_name in models_dict:
        model_dict = models_dict[model_name]
        model = model_dict[MODEL_KEY]
        preproc_garmin_data = model_dict[PREPROC_GARMIN_DATA_KEY]
        preproc_activity = model_dict[PREPROC_ACTIVITY_KEY]

        return preproc_garmin_data, preproc_activity, model
    else:
        raise HTTPException(
            status_code=404, detail=f"Model {model_name} not found")


@tomorrow_app.get("/")
def root():
    return {"Overcome": "Tomorrow"}
