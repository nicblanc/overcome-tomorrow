import numpy as np
import pandas as pd

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
        df["start_sleep"] = df["start_sleep"].dt.hour.astype(float) / 24
        df["end_sleep"] = df["end_sleep"].dt.hour.astype(float) / 24
        df["beginTimestamp"] = df["beginTimestamp"].dt.hour.astype(float) / 24
        df["start_sleep_sin"] = np.sin(2 * np.pi * df["start_sleep"])
        df["start_sleep_cos"] = np.cos(2 * np.pi * df["start_sleep"])
        df["end_sleep_sin"] = np.sin(2 * np.pi * df["end_sleep"])
        df["end_sleep_cos"] = np.cos(2 * np.pi * df["end_sleep"])
        df["beginTimestamp_sin"] = np.sin(2 * np.pi * df["beginTimestamp"])
        df["beginTimestamp_cos"] = np.cos(2 * np.pi * df["beginTimestamp"])
        df = df.drop(columns=["start_sleep", "end_sleep", "beginTimestamp"])
        return df

    def get_feature_names_out(self):
        return self.columns


class CyclicalFeaturesActivity(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"].dt.hour.astype(float) / 24
        df["start_time"] = pd.to_datetime(df["start_time"])
        df["start_time"] = df["start_time"].dt.hour.astype(float) / 24
        df["timestamp_sin"] = np.sin(2 * np.pi * df["timestamp"])
        df["timestamp_cos"] = np.cos(2 * np.pi * df["timestamp"])
        df["start_time_sin"] = np.sin(2 * np.pi * df["start_time"])
        df["start_time_cos"] = np.cos(2 * np.pi * df["start_time"])
        df = df.drop(columns=["timestamp", "start_time"])
        return df

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


def preproc_garmin_data(data=None) -> pd.DataFrame:
    """
    preprocessing of datas garmin
    data : by default None
    """
    # CHECK DATA
    if data is None:
        data = merge_all_data("../../raw_data/Wellness/",
                              "../../raw_data/Fitness/", "../../raw_data/Aggregator/")

    full_preprocessing = create_preproc_garmin_data(data)
    data_preprocess = full_preprocessing.fit_transform(data)
    print("✅ Preprocess successful")
    return data_preprocess


def create_preproc_activity(activity_df):

    # get numerical data
    data_num = activity_df.select_dtypes(include=np.number)
    data_num

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
        ('knn_imputer', KNNImputer(n_neighbors=4)),
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

    full_proces = ColumnTransformer(transformers=[
        ("knn_imputer", pipe_knn_imputer, numerical_features),
        ("imputer 100", pipe_nan_to_100, beginning_stamina_features),
        ("cat_encoder", pipe_onehot, ["sport"]),
        ("cycle_encoder", CyclicalFeaturesActivity(),
         ["timestamp", "start_time"])
    ]).set_output(transform="pandas")
    return full_proces


def preproc_activity(activity_df=None) -> pd.DataFrame:
    """
    preprocessing of data activity
    activity_df : by default None
    """
    if activity_df is None:
        print("❌ No data")
        return None
    full_proces = create_preproc_activity(activity_df)
    data_preprocessed = full_proces.fit_transform(activity_df)
    print("✅ Preprocess successful")
    return data_preprocessed
