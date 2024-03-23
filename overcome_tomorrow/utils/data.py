import pandas as pd
import pathlib
import os

from datetime import datetime
from google.cloud import bigquery, storage
from overcome_tomorrow.params import *

from os.path import join

######################### LOADING FUNCTION ####################################


def loading_data(path):
    """
    Loading dataset from a specific path
    """
    dataframes = []
    for fichier in os.listdir(path):
        complet_path = join(path, fichier)
        if fichier.endswith('.json'):
            df_temp = pd.read_json(complet_path)
            dataframes.append(df_temp)

    df_final = pd.concat(dataframes, ignore_index=True)

    print(f"✅ {path} loaded")
    return df_final


def load_fitness_data(path):
    """
    Loading Fitness dataset
    """
    dataframes = []
    for fichier in os.listdir(path):
        complet_path = join(path, fichier)
        if fichier.endswith('.json'):
            df_initial = pd.read_json(complet_path)
            if 'summarizedActivitiesExport' in df_initial.columns:
                df_exploded = df_initial['summarizedActivitiesExport'].explode(
                )
                df_normalized_list = [pd.json_normalize(
                    x) for x in df_exploded if x is not None]
                df_final = pd.concat(df_normalized_list, ignore_index=True)
            dataframes.append(df_final)
    df = pd.concat(dataframes, ignore_index=True)

    print(f"✅ {path} loaded")
    return df

######################### CLEANING FUNCTIONS ###################################


def clean_data_aggregator(path="raw_data/Aggregator") -> pd.DataFrame:
    """
    Clean aggregator's dataset
        - assigning correct dtypes to each column
        - removing useless columns
        - deal with NaN
    """
    df = loading_data(path)
    df['calendarDate'] = pd.to_datetime(df['calendarDate'])
    df = df.sort_values(by='calendarDate', ascending=True)
    df = df.loc[df['calendarDate'] >= pd.Timestamp('2019-01-01')]
    df = df.drop_duplicates(subset=['calendarDate'], keep='first')
    df.set_index('calendarDate', inplace=True)

    # Récupération des valeurs intéressantes dans la colonne hydration pour chaque date
    df['hydration_sweatLossInML'] = df['hydration'].apply(
        lambda x: x.get('sweatLossInML') if isinstance(x, dict) else None)

    # Same for Respiration
    df['respiration_avgWakingRespirationValue'] = df['respiration'].apply(
        lambda x: x.get('avgWakingRespirationValue') if isinstance(x, dict) else None)
    df['respiration_highestRespirationValue'] = df['respiration'].apply(
        lambda x: x.get('highestRespirationValue') if isinstance(x, dict) else None)
    df['respiration_lowestRespirationValue'] = df['respiration'].apply(
        lambda x: x.get('lowestRespirationValue') if isinstance(x, dict) else None)

    # Same for bodyBattery
    df['bodyBattery_Highest'] = df['bodyBattery'].apply(lambda x: x.get('bodyBatteryStatList')[0].get('statsValue') if isinstance(
        x, dict) and isinstance(x.get('bodyBatteryStatList'), list) and len(x.get('bodyBatteryStatList')) > 0 else None)
    df['bodyBattery_Lowest'] = df['bodyBattery'].apply(lambda x: x.get('bodyBatteryStatList')[1].get('statsValue') if isinstance(
        x, dict) and isinstance(x.get('bodyBatteryStatList'), list) and len(x.get('bodyBatteryStatList')) > 0 else None)

    # Same for allDayStress
    df['allDayStress_awake'] = df['allDayStress'].apply(lambda x: x.get('aggregatorList')[1].get('averageStressLevel') if isinstance(
        x, dict) and isinstance(x.get('aggregatorList'), list) and len(x.get('aggregatorList')) > 0 else None)
    df['allDayStress_asleep'] = df['allDayStress'].apply(lambda x: x.get('aggregatorList')[2].get('averageStressLevel') if isinstance(
        x, dict) and isinstance(x.get('aggregatorList'), list) and len(x.get('aggregatorList')) > 0 else None)

    features_to_drop = ['userProfilePK',
                        'uuid', 'durationInMilliseconds',
                        'dailyStepGoal',
                        'netCalorieGoal',
                        'wellnessStartTimeGmt',
                        'wellnessEndTimeGmt',
                        'userIntensityMinutesGoal',
                        'userFloorsAscendedGoal',
                        'includesWellnessData',
                        'includesActivityData', 'includesCalorieConsumedData',
                        'includesSingleMeasurement', 'includesContinuousMeasurement',
                        'includesAllDayPulseOx', 'includesSleepPulseOx', 'source',
                        'version', 'isVigorousDay',
                        'restingHeartRateTimestamp',
                        'burnedKilocalories', 'consumedKilocalories', 'wellnessStartTimeLocal', 'wellnessEndTimeLocal',
                        'moderateIntensityMinutes', 'vigorousIntensityMinutes', 'averageMonitoringEnvironmentAltitude',
                        # Features extraites
                        'respiration', 'bodyBattery', 'hydration', 'allDayStress', 'bodyBatteryFeedback',
                        # Doublon entre les calories "Wellness"
                        'wellnessKilocalories', 'remainingKilocalories', 'wellnessTotalKilocalories', 'wellnessActiveKilocalories',
                        'bmrKilocalories',
                        # Data leakage
                        'totalSteps', 'minHeartRate', 'maxHeartRate', 'currentDayRestingHeartRate',
                        # Colonne que des NaN qui provient surement de donnée de 2006 qui sont plus dans les datasets 2024
                        'averageSpo2Value', 'lowestSpo2Value', 'latestSpo2Value', 'latestSpo2ValueReadingTimeGmt', 'latestSpo2ValueReadingTimeLocal',
                        'restingCaloriesFromActivity', 'totalPushes', 'pushDistance', 'jetLagDay', 'jetLagTripName', 'jetLagTripPk', 'dailyTotalFromEpochData']

    aggregator = df.drop(columns=features_to_drop)
    print("✅ Aggregator cleaned")

    return aggregator


def clean_wellness(path="raw_data/Wellness") -> pd.DataFrame:
    sleep_df = loading_data(path)

    # DELETE DUPLICATE
    sleep_df = sleep_df.drop_duplicates(
        subset=["calendarDate", "sleepStartTimestampGMT", "sleepEndTimestampGMT"])

    # DELETE USELESS FEATURE
    useless_cols = ["sleepResultType",
                    "napList",
                    "sleepWindowConfirmationType",
                    "retro", "spo2SleepSummary",
                    "averageRespiration",
                    "lowestRespiration",
                    "highestRespiration",
                    "restlessMomentCount",
                    "awakeCount",
                    "unmeasurableSeconds",
                    "avgSleepStress"]

    clean_sleep_df = sleep_df.drop(columns=useless_cols)

    # CONVERT DATETIME TYPE
    clean_sleep_df["date"] = pd.to_datetime(clean_sleep_df["calendarDate"])
    clean_sleep_df["start_sleep"] = pd.to_datetime(
        clean_sleep_df["sleepStartTimestampGMT"])
    clean_sleep_df["end_sleep"] = pd.to_datetime(
        clean_sleep_df["sleepEndTimestampGMT"])
    clean_sleep_df.drop(columns=["sleepStartTimestampGMT",
                        "sleepEndTimestampGMT", "calendarDate"], inplace=True)

    # REPLACE COLUMNS
    cols = ["date", "start_sleep", "end_sleep", "deepSleepSeconds",
            "lightSleepSeconds", "remSleepSeconds", "awakeSleepSeconds", "sleepScores"]
    clean_sleep_df = clean_sleep_df[cols]

    # GET QUALITY SCORE
    clean_sleep_df_copy = clean_sleep_df.copy()
    clean_sleep_df_copy['qualityScore'] = clean_sleep_df_copy['sleepScores'].apply(
        lambda x: x.get('qualityScore') if isinstance(x, dict) else None)
    clean_sleep_df_copy.drop(columns="sleepScores", inplace=True)
    print("✅ Wellness cleaned")
    return clean_sleep_df_copy


def clean_fitness(path="raw_data/Fitness") -> pd.DataFrame:
    fitness_df = load_fitness_data(path)

    # DELETE DUPLICATE
    fitness_df.drop_duplicates(subset=["beginTimestamp"])

    # SELECT FEATURE
    ALL_KEYS = [
        'beginTimestamp',
        'activityTrainingLoad',
        'activityType',
        'aerobicTrainingEffect',
        'aerobicTrainingEffectMessage',
        'anaerobicTrainingEffect',
        'anaerobicTrainingEffectMessage',
        'avgBikeCadence',
        'avgHr',
        'avgPower',
        'avgRunCadence',
        'avgSpeed',
        'calories',
        'caloriesConsumed',
        'distance',
        'duration',
        'maxHr',
        'maxPower',
        'maxRunCadence',
        'maxSpeed',
        'moderateIntensityMinutes',
        'normPower',
        'sportType',
        'trainingEffectLabel',
        'trainingStressScore',
        'vigorousIntensityMinutes',
    ]
    clean_fitness_df = fitness_df[ALL_KEYS]
    # Convert datetime
    clean_fitness_df["date"] = pd.to_datetime(
        clean_fitness_df["beginTimestamp"], unit="ms").dt.date
    clean_fitness_df["date"] = pd.to_datetime(clean_fitness_df["date"])
    clean_fitness_df["beginTimestamp"] = pd.to_datetime(
        clean_fitness_df["beginTimestamp"], unit="ms")

    print("✅ Fitness cleaned")
    return clean_fitness_df

############################ MERGING FUNCTIONS #################################


def merge_all_data(sleep_path="raw_data/Wellness",
                   fitness_path="raw_data/Fitness",
                   aggregator_path="raw_data/Aggregator") -> pd.DataFrame:

    sleep_df = clean_wellness(sleep_path)
    fitness_df = clean_fitness(fitness_path)
    aggregator = clean_data_aggregator(aggregator_path)

    sleep_fitness = pd.merge(sleep_df, fitness_df, on="date", how="left")
    sleep_fitness = sleep_fitness.rename(columns={'date': 'calendarDate'})
    sleep_fitness = sleep_fitness.sort_values(
        by='calendarDate', ascending=True)
    sleep_fitness.set_index('calendarDate', inplace=True)

    final_dataset = pd.merge(sleep_fitness, aggregator,
                             left_index=True, right_index=True, how='left')

    print("✅ Datasets merged")
    return final_dataset


def load_to_csv(output_path="raw_data/"):
    complet_path = join(output_path, "fitness_data.csv")
    final_df = merge_all_data()
    final_df.to_csv(complet_path, index=True)
    print("✅ Save as csv")


######################## GOOGLE BIGQUERY FUNCTIONS #############################


def upload_dataframe_to_bq(
        data: pd.DataFrame,
        table: str,
        gcp_project: str = GCP_PROJECT,
        bq_dataset: str = BQ_DATASET):

    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

    """ Create dataset if not exists """
    client.create_dataset(dataset=bq_dataset, exists_ok=True)

    """ Upload dataset to Google BigQuery """
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(f"\n⌛ Saving data to BigQuery @ {full_table_name}... ⌛")

    print(f"\nWrite {full_table_name} ({data.shape[0]} rows)")
    job = client.load_table_from_dataframe(
        data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")


def upload_csv_to_bq(
        path: str,
        gcp_project: str = GCP_PROJECT,
        bq_dataset: str = BQ_DATASET):
    """ Upload csv to Google BigQuery """
    dataframe = pd.read_csv(path)
    table = pathlib.PurePath(path).stem
    upload_dataframe_to_bq(dataframe, table, gcp_project, bq_dataset)


def load_csv_from_bq(types,
                     table: str,
                     gcp_project: str = GCP_PROJECT,
                     bq_dataset: str = BQ_DATASET):

    print(f"⌛ Loading {table} data from BigQuery server... ⌛ ")

    query = f"""
        SELECT *
        FROM {gcp_project}.{bq_dataset}.{table}
    """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    df = df.astype(types)
    print(f"✅ {table} data loaded, with shape {df.shape}")
    return df


def save_csv_from_bq(path: str,
                     table: str,
                     gcp_project: str = GCP_PROJECT,
                     bq_dataset: str = BQ_DATASET):
    df = load_csv_from_bq(table, gcp_project, bq_dataset)
    df.to_csv(path)


###################### GOOGLE CLOUD STORAGE FUNCTIONS ##########################


def upload_model_to_gcs(model_path: str = join(MODEL_PATH, MODEL_NAME),
                        bucket_name: str = BUCKET_NAME,
                        gcp_project: str = GCP_PROJECT,
                        location: str = GCP_REGION):
    model_filename = pathlib.PurePath(model_path).name
    client = storage.Client()
    """ Create bucket if not exists """
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        bucket = client.create_bucket(
            bucket_name, project=gcp_project, location=location)

    """ Upload model file to blob """
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)
    print("✅ Model saved to GCS")


def upload_preprocessors_to_gcs(preprocessors_path: str = MODEL_PATH,
                                bucket_name: str = BUCKET_NAME,
                                gcp_project: str = GCP_PROJECT,
                                location: str = GCP_REGION):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    """ Create bucket if not exists """
    if not bucket.exists():
        bucket = client.create_bucket(
            bucket_name, project=gcp_project, location=location)

    preproc_garmin_data_path = join(
        preprocessors_path, GARMIN_DATA_PREPROC_NAME)
    preproc_garmin_data_filename = pathlib.PurePath(
        preproc_garmin_data_path).name

    preproc_activity_path = join(preprocessors_path, ACTIVITY_PREPROC_NAME)
    preproc_activity_filename = pathlib.PurePath(preproc_activity_path).name

    """ Upload preprocessors files to blob """
    preproc_garmin_data_blob = bucket.blob(
        f"models/{preproc_garmin_data_filename}")
    preproc_garmin_data_blob.upload_from_filename(preproc_garmin_data_path)
    preproc_activity_blob = bucket.blob(f"models/{preproc_activity_filename}")
    preproc_activity_blob.upload_from_filename(preproc_activity_path)

    print("✅ Preprocessors saved to GCS")


def get_model_and_preprocessors_blobs_from_gcs(model_filename: str = MODEL_NAME,
                                               preproc_garmin_data_filename: str = GARMIN_DATA_PREPROC_NAME,
                                               preproc_activity_filename: str = ACTIVITY_PREPROC_NAME,
                                               bucket_name: str = BUCKET_NAME):

    client = None
    try:
        client = storage.Client()
    except Exception as e:
        print(
            f"\n⚠️ Cannot connect to Google Cloud Storage ⚠️\nFollowing error occured:\n{e}")
        return None, None, None

    bucket = client.get_bucket(bucket_name)
    if not bucket.exists():
        print(f"\n❌ Bucket {bucket_name} not found")
        return None, None, None

    preproc_garmin_data_blob = bucket.blob(
        f"models/{preproc_garmin_data_filename}")
    preproc_activity_blob = bucket.blob(f"models/{preproc_activity_filename}")
    model_blob = bucket.blob(f"models/{model_filename}")

    if not preproc_garmin_data_blob.exists():
        print(f"\n❌ Blob {preproc_garmin_data_filename} not found")
        return None, None, None
    if not preproc_activity_blob.exists():
        print(f"\n❌ Blob {preproc_activity_filename} not found")
        return None, None, None
    if not model_blob.exists():
        print(f"\n❌ Blob {model_filename} not found")
        return None, None, None

    return model_blob, preproc_garmin_data_blob, preproc_activity_blob


def get_last_modified_dates_for_model_and_preprocessors_from_gcs(model_filename: str = MODEL_NAME,
                                                                 preproc_garmin_data_filename: str = GARMIN_DATA_PREPROC_NAME,
                                                                 preproc_activity_filename: str = ACTIVITY_PREPROC_NAME,
                                                                 bucket_name: str = BUCKET_NAME):

    model_blob, preproc_garmin_data_blob, preproc_activity_blob = get_model_and_preprocessors_blobs_from_gcs(
        model_filename, preproc_garmin_data_filename, preproc_activity_filename, bucket_name)

    if (model_blob is None) or (preproc_garmin_data_blob is None) or (preproc_activity_blob is None):
        default_date = datetime(1970, 1, 1)
        return default_date, default_date, default_date

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    model_blob_updated = bucket.get_blob(model_blob.name).updated
    preproc_garmin_data_blob_updated = bucket.get_blob(
        preproc_garmin_data_blob.name).updated
    preproc_activity_blob_updated = bucket.get_blob(
        preproc_activity_blob.name).updated
    return model_blob_updated, preproc_garmin_data_blob_updated, preproc_activity_blob_updated


def download_model_and_preprocessors_from_gcs(
        model_path: str = MODEL_PATH,
        model_filename: str = MODEL_NAME,
        preprocessors_path: str = MODEL_PATH,
        preproc_garmin_data_filename: str = GARMIN_DATA_PREPROC_NAME,
        preproc_activity_filename: str = ACTIVITY_PREPROC_NAME,
        bucket_name: str = BUCKET_NAME):

    model_blob, preproc_garmin_data_blob, preproc_activity_blob = get_model_and_preprocessors_blobs_from_gcs(
        model_filename, preproc_garmin_data_filename, preproc_activity_filename, bucket_name)
    if (model_blob is None) or (preproc_garmin_data_blob is None) or (preproc_activity_blob is None):
        return

    model_blob.download_to_filename(join(
        model_path, model_filename))
    print("✅ Latest model downloaded from cloud storage")
    preproc_activity_blob.download_to_filename(join(
        preprocessors_path, preproc_activity_filename))
    preproc_garmin_data_blob.download_to_filename(join(
        preprocessors_path, preproc_garmin_data_filename))
    print("✅ Latest preprocessors downloaded from cloud storage")
