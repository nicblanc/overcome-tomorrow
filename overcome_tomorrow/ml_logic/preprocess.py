import numpy as np
import pandas as pd
import math

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import TransformerMixin, BaseEstimator
from overcome_tomorrow.utils.data import *


class CyclicalFeaturesSleep(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df["start_sleep_sin"], df["start_sleep_cos"] = convert_time_to_sin_cos(
            df["start_sleep"])
        df["end_sleep_sin"], df["end_sleep_cos"] = convert_time_to_sin_cos(
            df["end_sleep"])
        df["beginTimestamp_sin"], df["beginTimestamp_cos"] = convert_time_to_sin_cos(
            df["beginTimestamp"])

        df = df.drop(columns=["start_sleep", "end_sleep", "beginTimestamp"])
        return df

    def get_feature_names_out(self):
        return self.columns


def convert_time_to_sin_cos(time):
    t = (time.dt.hour * 60 + time.dt.minute) / 1440
    s = np.sin(2 * np.pi * t)
    c = np.cos(2 * np.pi * t)
    return s, c


def convert_sin_cos_to_hour(s, c):
    angle = math.atan2(s, c)
    angle *= 180 / math.pi
    if angle < 0:
        angle += 360

    time = angle * 24.0 / 360.0
    hours = int(time)
    minutes = (time*60.0) % 60.0
    seconds = (time*3600.0) % 60.0

    return ("%d:%02d:%02d" % (hours, minutes, seconds))


class CyclicalFeaturesActivity(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):

        df["timestamp_sin"], df["timestamp_cos"] = convert_time_to_sin_cos(
            pd.to_datetime(df["timestamp"]))
        df["start_time_sin"], df["start_time_cos"] = convert_time_to_sin_cos(
            pd.to_datetime(df["start_time"]))
        df = df.drop(columns=["timestamp", "start_time"])
        return df

    def inverse_transform(self, target):
        res = []
        for row in target:
            start_time = convert_sin_cos_to_hour(row[2], row[3])
            timestamp = convert_sin_cos_to_hour(row[0], row[1])
            res.append((timestamp, start_time))
        return res

    def get_feature_names_out(self):
        return self.columns


def create_preproc_garmin_data(data):

    # Delete na for columns beginTimestam
    cycle_data = data.select_dtypes(include=np.datetime64).dropna()

    # get all cols for each type
    cycle_features = data.select_dtypes(include=np.datetime64).columns
    numerical_features = data.select_dtypes(include=np.number).columns

    # pipeline numerical features
    pipe_numerical_knn = Pipeline([
        ('knn_imputer', KNNImputer(n_neighbors=4)),
        ('robust_scaler', RobustScaler())
    ])

    # pipeline categorial features
    pipe_categorical = Pipeline([
        ("simple_imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(sparse_output=False,
         handle_unknown="ignore", drop="if_binary"))
    ])

    # full preprocessing
    full_preprocessing = ColumnTransformer(transformers=[
        ("knn_transformer", pipe_numerical_knn, numerical_features),
        ("pipe_categorical", pipe_categorical,
         ["sportType", "trainingEffectLabel"]),
        ("cycle transform", CyclicalFeaturesSleep(), cycle_features)
    ]).set_output(transform="pandas")
    return full_preprocessing


def create_preproc_activity(activity_df):
    # If all values are NA, drop that column.
    activity_df.dropna(axis=1, how="all", inplace=True)

    # get cols numerical
    numerical_features = activity_df.select_dtypes(include=np.number).columns
    numerical_features = set(numerical_features) - set(["avg_stroke_distance",
                                                        "num_active_lengths",
                                                        "num_lengths",
                                                        "max_running_cadence",
                                                        "pool_length",
                                                        "max_cadence",
                                                        "avg_step_length",
                                                        "normalized_power",
                                                        "avg_power",
                                                        "total_strokes",
                                                        "training_stress_score",
                                                        "avg_cadence",
                                                        "max_power"
                                                        ])
    numerical_features = list(numerical_features)

    # Stamina features
    # 205: beginning
    # 206: ending
    # 207: min
    beginning_stamina_features = ["205"]
    numerical_features = list(
        set(numerical_features) - set(beginning_stamina_features))

    # pipeline knn imputer
    pipe_knn_imputer = Pipeline([
        ('knn_imputer', KNNImputer(n_neighbors=4, add_indicator=True)),
        ('robust_scaler', RobustScaler())
    ])

    # pipeline simple imputer fill value with 100 for beginning stamina (205)
    pipe_nan_to_100 = Pipeline([
        ('simple_imputer', SimpleImputer(strategy="constant", fill_value=100)),
        ('robust_scaler', RobustScaler())
    ])

    # get data and cols categorial feature
    data_cat = activity_df.select_dtypes(exclude=np.number)
    cat_features = data_cat.columns
    cat_features = list(set(cat_features) - set(["pool_length_unit"]))

    # pipeline categorical

    pipe_onehot = Pipeline([
        ("simple_imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(sparse_output=False,
         handle_unknown="ignore", drop="if_binary"))
    ])

    full_proces = InvertableColumnTransformer(transformers=[
        ("knn_imputer", pipe_knn_imputer, numerical_features),
        ("cat_encoder", pipe_onehot, ["sport"]),
        ("cycle_encoder", CyclicalFeaturesActivity(),
         ["timestamp", "start_time"]),
        ("imputer 100", pipe_nan_to_100, beginning_stamina_features)
    ]).set_output(transform="pandas")

    return full_proces


class InvertableColumnTransformer(ColumnTransformer):
    """
    Adds an inverse transform method to the standard sklearn.compose.ColumnTransformer.
    """

    def inverse_transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        transformer_to_cols = {}
        out = {}
        for t in self.transformers_:
            name = t[0]
            cols = t[-1]
            if name not in ("remainder"):
                transformer_to_cols[name] = cols
                for col in cols:
                    out[col] = []

        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            start = indices.start
            stop = indices.stop
            arr = X[:, start: stop]
            if transformer in (None, "remainder", "passthrough", "drop"):
                pass

            else:
                try:
                    if isinstance(transformer, Pipeline):
                        tmp = transformer[1].inverse_transform(arr)
                        current_cols = transformer_to_cols[name]
                        for sub in tmp:
                            for col, val in zip(current_cols, sub):
                                out[col].append(val)
                    else:
                        tmp = transformer.inverse_transform(arr)
                        current_cols = transformer_to_cols[name]
                        for sub in tmp:
                            for col, val in zip(current_cols, sub):
                                out[col].append(val)
                except Exception as e:
                    print(e)
        return pd.DataFrame.from_dict(out)
