import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from overcome_tomorrow.utils.data import *
from overcome_tomorrow.ml_logic.preprocess import *
from tqdm import tqdm


def get_data():
    # TODO load from Google Cloud
    garmin_data = merge_all_data(
        "../raw_data/Wellness/", "../raw_data/Fitness/", "../raw_data/Aggregator/")
    activities = pd.read_csv("../raw_data/activities.csv",
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
        # for i in range(steps, steps + 5):
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


def create_model(X_train, y_train):

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


def create_train_and_save_model():
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
    model.fit(X_train, y_train, batch_size=32, epochs=epochs)
    model.summary()
    model.save("first_model.keras")
