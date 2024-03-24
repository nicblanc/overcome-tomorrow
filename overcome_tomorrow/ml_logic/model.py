import numpy as np
import pandas as pd
from overcome_tomorrow.utils.data import *
from overcome_tomorrow.ml_logic.preprocess import *
from overcome_tomorrow.params import *
from os import makedirs
from os.path import join, exists
from tqdm import tqdm
from pickle import dump, load
from datetime import datetime, timedelta

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping


def get_data(input_path=DATA_PATH):
    try:
        activities = load_csv_from_bq(DTYPES_ACTIVITIES_RAW, "activities")
        garmin_data = load_csv_from_bq(DTYPES_GARMIN_DATA_RAW, "garmin_data")

        garmin_data.sort_values(by=["beginTimestamp"], inplace=True)
        garmin_data.dropna(subset=["beginTimestamp"], inplace=True)
        garmin_data.reset_index(drop=True, inplace=True)
        activities.sort_values(by=["start_time"], inplace=True)
        activities.reset_index(drop=True, inplace=True)

        return garmin_data, activities
    except Exception as e:
        print(
            f"⚠️ Trying to load data locally ⚠️\n Following error occured during loading data from BigQuery:\n{e}")

        garmin_data_path = join(DATA_PATH, "garmin_data.csv")
        activities_path = join(DATA_PATH, "activities.csv")
        garmin_data = None

        try:
            garmin_data = pd.read_csv(garmin_data_path, parse_dates=[
                                      "start_sleep", "end_sleep", "beginTimestamp"])
        except Exception:
            wellness_path = join(DATA_PATH, "Wellness/")
            fitness_path = join(DATA_PATH, "Fitness/")
            aggregator_path = join(DATA_PATH, "Aggregator/")
            garmin_data = merge_all_data(
                wellness_path, fitness_path, aggregator_path)

        activities = pd.read_csv(activities_path,
                                 parse_dates=["timestamp", "start_time"])
        garmin_data.sort_values(by=["beginTimestamp"], inplace=True)
        garmin_data.dropna(subset=["beginTimestamp"], inplace=True)
        activities.sort_values(by=["start_time"], inplace=True)
        activities.reset_index(drop=True, inplace=True)
        return garmin_data, activities


def create_sliding_windows_dataset(garmin_data, activities, preproc_garmin_data, preproc_activity):
    days = 30
    avg_activities_per_day = 2
    steps = days * avg_activities_per_day  # sliding window size
    # Prepare the training data
    X_train = []
    y_train = []

    for i in tqdm(range(steps, activities.shape[0]), desc="⌛ Creating Sliding Window dataset... ⌛"):
        activity = activities.iloc[[i]]
        activity_time = activity["start_time"][i].strftime('%Y-%m-%d %H:%M:%S')
        window_df = garmin_data[garmin_data["beginTimestamp"]
                                < activity_time].iloc[i - steps:i]
        # TODO find a way to preprocess everything only once and then create the sliding windows dataset
        X_train.append(preproc_garmin_data.transform(window_df))
        # TODO y_train.append(preproc_activity.transform(activity)[0])
        y_train.append(preproc_activity.transform(activity))

    X_train, y_train = np.array(X_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[2]))
    return X_train, y_train


def get_sliding_windows_for_n_last_days(garmin_data, preproc_garmin_data, last_days=30):
    days = 30
    avg_activities_per_day = 2
    steps = days * avg_activities_per_day  # sliding window size
    sliding_windows = []
    last_date = garmin_data.iloc[-1]["beginTimestamp"]
    for i in range(last_days):
        delta = timedelta(days=i)
        date = (last_date - delta)
        date = date.strftime('%Y-%m-%d %H:%M:%S')
        window_df = garmin_data[garmin_data["beginTimestamp"]
                                < date].iloc[-steps - 1:-1]
        sliding_windows.append(preproc_garmin_data.transform(window_df))
    return np.array(sliding_windows)


def get_sliding_window_for_date(garmin_data, date=datetime.now()):
    days = 30
    avg_activities_per_day = 2
    steps = days * avg_activities_per_day  # sliding window size
    date = date.strftime('%Y-%m-%d %H:%M:%S')
    window_df = garmin_data[garmin_data["beginTimestamp"]
                            < date].iloc[-steps - 1:-1]
    return window_df


def create_model(X_train, y_train):
    # TODO implement a real model, not just random layers :D
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=y_train.shape[1]))
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    return model


def create_train_and_save_model(model_path: str = MODEL_PATH,
                                model_filename: str = MODEL_NAME,
                                preprocessors_path: str = MODEL_PATH,
                                preproc_garmin_data_filename: str = GARMIN_DATA_PREPROC_NAME,
                                preproc_activity_filename: str = ACTIVITY_PREPROC_NAME):
    garmin_data, activities = get_data()

    # Fit Preprocessors
    preproc_garmin_data = create_preproc_garmin_data(garmin_data)
    preproc_activity = create_preproc_activity(activities)
    preproc_garmin_data.fit(garmin_data)
    preproc_activity.fit(activities)

    # Create sliding windows
    X_train, y_train = create_sliding_windows_dataset(
        garmin_data, activities, preproc_garmin_data, preproc_activity)

    # Create model
    epochs = 100
    model = create_model(X_train, y_train)
    # TODO train test split + validation data
    model.fit(X_train, y_train, batch_size=32, epochs=epochs)
    model.summary()

    model_name = pathlib.PurePath(model_filename).stem
    full_model_path = join(model_path, model_name, model_filename)
    model_parent = pathlib.PurePath(full_model_path).parent.as_posix()
    if not exists(model_parent):
        makedirs(model_parent)

    full_preprocessors_path = join(preprocessors_path, model_name)
    if not exists(full_preprocessors_path):
        makedirs(full_preprocessors_path)

    model.save(full_model_path)
    dump(preproc_garmin_data, open(
        join(preprocessors_path, model_name, preproc_garmin_data_filename), "wb"))
    dump(preproc_activity, open(
        join(preprocessors_path, model_name, preproc_activity_filename), "wb"))

    try:
        upload_model_to_gcs(model_path=full_model_path)
        upload_preprocessors_to_gcs(
            preprocessors_path=preprocessors_path, model_name=model_name)
    except Exception as e:
        print(
            f"\n⚠️ Cannot upload model/preprocessors to Google Cloud Storage ⚠️\nFollowing error occured:\n{e}")


def predict_for_date(garmin_data, preproc_garmin_data, preproc_activity, model, date=datetime.now()):
    window_df = get_sliding_window_for_date(garmin_data, date)
    input = np.array([preproc_garmin_data.transform(window_df)])
    prediction = model.predict(input)
    return preproc_activity.inverse_transform(prediction)


def predict_vs_real_for_date(garmin_data, activities, preproc_garmin_data, preproc_activity, model, date=datetime.now()):
    prediction = predict_for_date(
        garmin_data, preproc_garmin_data, preproc_activity, model, date)
    reals = activities[activities["start_time"].dt.strftime(
        '%Y-%m-%d %H:%M:%S') >= date.strftime('%Y-%m-%d %H:%M:%S')]
    if len(reals) > 0:
        return pd.concat([prediction, reals.iloc[[0]]],  keys=['prediction', "real"])
    return pd.concat([prediction],  keys=["prediction"])


def predict_for_last_n_days(garmin_data, preproc_garmin_data, preproc_activity, model, last_days=30):
    input = get_sliding_windows_for_n_last_days(
        garmin_data, preproc_garmin_data, last_days)
    predictions = model.predict(input)
    return preproc_activity.inverse_transform(predictions)


def predict_vs_real_for_last_n_days(garmin_data, activities, preproc_garmin_data, preproc_activity, model, last_days=30):
    # input = get_sliding_windows_for_n_last_days(
    #     garmin_data, preproc_garmin_data, last_days)
    # predictions = preproc_activity.inverse_transform(model.predict(input))

    delta = timedelta(days=last_days)
    last_date = (garmin_data.iloc[-1]["beginTimestamp"] -
                 delta).strftime('%Y-%m-%d %H:%M:%S')
    date = garmin_data[garmin_data["beginTimestamp"]
                       < last_date].iloc[-1]["beginTimestamp"]

    predictions = predict_for_last_n_days(
        garmin_data, preproc_garmin_data, preproc_activity, model, last_days)

    reals = activities[activities["start_time"].dt.strftime(
        '%Y-%m-%d %H:%M:%S') >= date.strftime('%Y-%m-%d %H:%M:%S')]
    size = min(len(predictions), len(reals))
    return pd.concat([predictions, reals.iloc[0:size]],  keys=['predictions', "reals"])
