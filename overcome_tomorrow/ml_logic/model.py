import numpy as np
import pandas as pd
from overcome_tomorrow.utils.data import *
from overcome_tomorrow.ml_logic.preprocess import *
from overcome_tomorrow.params import *
from os.path import isfile, join, exists
from tqdm import tqdm
from pickle import dump, load
from datetime import datetime, timedelta

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping


def get_data(input_path=DATA_PATH):
    # TODO load from Google Cloud
    wellness_path = join(DATA_PATH, "Wellness/")
    fitness_path = join(DATA_PATH, "Fitness/")
    aggregator_path = join(DATA_PATH, "Aggregator/")
    activities_path = join(DATA_PATH, "activities.csv")
    garmin_data = merge_all_data(wellness_path, fitness_path, aggregator_path)
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

    for i in tqdm(range(steps, activities.shape[0])):
        activity = activities.iloc[[i]]
        activity_time = activity["start_time"][i].strftime('%Y-%m-%d %H:%M:%S')
        window_df = garmin_data[garmin_data["beginTimestamp"]
                                < activity_time].iloc[i - steps:i]
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


def create_train_and_save_model(output_path="."):
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
    model.save(join(output_path, "first_model.keras"))
    dump(preproc_garmin_data, open(
        join(output_path, "preproc_garmin_data.pkl"), "wb"))
    dump(preproc_activity, open(join(output_path, "preproc_activity.pkl"), "wb"))


def load_preprocessors_and_model(path="."):
    preproc_garmin_data = load(
        open(join(path, "preproc_garmin_data.pkl"), "rb"))
    preproc_activity = load(open(join(path, "preproc_activity.pkl"), "rb"))
    model = load_model(join(path, "first_model.keras"))
    return preproc_garmin_data, preproc_activity, model


def predict_for_date(garmin_data, preproc_garmin_data, preproc_activity, model, date=datetime.now()):
    window_df = get_sliding_window_for_date(garmin_data, date)
    input = np.array([preproc_garmin_data.transform(window_df)])
    prediction = model.predict(input)
    return preproc_activity.inverse_transform(prediction)


def predict_for_last_n_days(garmin_data, preproc_garmin_data, preproc_activity, model, last_days=30):
    input = get_sliding_windows_for_n_last_days(
        garmin_data, preproc_garmin_data, last_days)
    prediction = model.predict(input)
    return preproc_activity.inverse_transform(prediction)
