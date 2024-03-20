import pandas as pd
import os

######################### LOADING FUNCTION ####################################

def loading_data(path):
    """
    Loading dataset from a specific path
    """
    dataframes = []
    for fichier in os.listdir(path):
        complet_path = os.path.join(path, fichier)
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
        complet_path = os.path.join(path, fichier)
        if fichier.endswith('.json'):
            df_initial = pd.read_json(complet_path)
            if 'summarizedActivitiesExport' in df_initial.columns:
                df_exploded = df_initial['summarizedActivitiesExport'].explode()
                df_normalized_list = [pd.json_normalize(x) for x in df_exploded if x is not None]
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
    df.set_index('calendarDate',inplace=True)

    # Récupération des valeurs intéressantes dans la colonne hydration pour chaque date
    df['hydration_sweatLossInML'] = df['hydration'].apply(lambda x: x.get('sweatLossInML') if isinstance(x, dict) else None)

    # Same for Respiration
    df['respiration_avgWakingRespirationValue'] = df['respiration'].apply(lambda x: x.get('avgWakingRespirationValue') if isinstance(x, dict) else None)
    df['respiration_highestRespirationValue'] = df['respiration'].apply(lambda x: x.get('highestRespirationValue') if isinstance(x, dict) else None)
    df['respiration_lowestRespirationValue'] = df['respiration'].apply(lambda x: x.get('lowestRespirationValue') if isinstance(x, dict) else None)

    # Same for bodyBattery
    df['bodyBattery_Highest'] = df['bodyBattery'].apply(lambda x: x.get('bodyBatteryStatList')[0].get('statsValue') if isinstance(x, dict) and isinstance(x.get('bodyBatteryStatList'), list) and len(x.get('bodyBatteryStatList')) > 0 else None)
    df['bodyBattery_Lowest'] = df['bodyBattery'].apply(lambda x: x.get('bodyBatteryStatList')[1].get('statsValue') if isinstance(x, dict) and isinstance(x.get('bodyBatteryStatList'), list) and len(x.get('bodyBatteryStatList')) > 0 else None)

    # Same for allDayStress
    df['allDayStress_awake'] = df['allDayStress'].apply(lambda x: x.get('aggregatorList')[1].get('averageStressLevel') if isinstance(x, dict) and isinstance(x.get('aggregatorList'), list) and len(x.get('aggregatorList')) > 0 else None)
    df['allDayStress_asleep'] = df['allDayStress'].apply(lambda x: x.get('aggregatorList')[2].get('averageStressLevel') if isinstance(x, dict) and isinstance(x.get('aggregatorList'), list) and len(x.get('aggregatorList')) > 0 else None)

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
                    'version','isVigorousDay',
                    'restingHeartRateTimestamp',
                    'burnedKilocalories', 'consumedKilocalories','wellnessStartTimeLocal','wellnessEndTimeLocal',
                    'moderateIntensityMinutes','vigorousIntensityMinutes','averageMonitoringEnvironmentAltitude',
                     ## Features extraites
                    'respiration','bodyBattery','hydration','allDayStress','bodyBatteryFeedback',
                     ## Doublon entre les calories "Wellness"
                    'wellnessKilocalories','remainingKilocalories','wellnessTotalKilocalories','wellnessActiveKilocalories',
                    'bmrKilocalories',
                     ## Data leakage
                    'totalSteps','minHeartRate','maxHeartRate','currentDayRestingHeartRate',
                     ## Colonne que des NaN qui provient surement de donnée de 2006 qui sont plus dans les datasets 2024
                    'averageSpo2Value','lowestSpo2Value','latestSpo2Value','latestSpo2ValueReadingTimeGmt','latestSpo2ValueReadingTimeLocal',
                    'restingCaloriesFromActivity','totalPushes','pushDistance','jetLagDay','jetLagTripName','jetLagTripPk','dailyTotalFromEpochData']

    aggregator = df.drop(columns=features_to_drop)
    print("✅ Aggregator cleaned")

    return aggregator

def clean_wellness(path="raw_data/Wellness") -> pd.DataFrame:
    sleep_df = loading_data(path)

    # DELETE DUPLICATE
    sleep_df = sleep_df.drop_duplicates(subset=["calendarDate", "sleepStartTimestampGMT", "sleepEndTimestampGMT"])

    # DELETE USELESS FEATURE
    useless_cols = ["sleepResultType",
                    "napList",
                    "sleepWindowConfirmationType",
                    "retro","spo2SleepSummary",
                    "averageRespiration",
                    "lowestRespiration",
                    "highestRespiration",
                    "restlessMomentCount",
                    "awakeCount",
                    "unmeasurableSeconds",
                    "avgSleepStress"]

    clean_sleep_df = sleep_df.drop(columns=useless_cols)

    # CONVERT DATETIME TYPE
    clean_sleep_df["date"]  = pd.to_datetime(clean_sleep_df["calendarDate"])
    clean_sleep_df["start_sleep"] = pd.to_datetime(clean_sleep_df["sleepStartTimestampGMT"])
    clean_sleep_df["end_sleep"] = pd.to_datetime(clean_sleep_df["sleepEndTimestampGMT"])
    clean_sleep_df.drop(columns=["sleepStartTimestampGMT", "sleepEndTimestampGMT", "calendarDate"], inplace=True)

    # REPLACE COLUMNS
    cols = ["date", "start_sleep", "end_sleep", "deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds", "awakeSleepSeconds", "sleepScores"]
    clean_sleep_df = clean_sleep_df[cols]

    # GET QUALITY SCORE
    clean_sleep_df_copy = clean_sleep_df.copy()
    clean_sleep_df_copy['qualityScore'] = clean_sleep_df_copy['sleepScores'].apply(lambda x: x.get('qualityScore') if isinstance(x, dict) else None)
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
    clean_fitness_df["date"] = pd.to_datetime(clean_fitness_df["beginTimestamp"], unit="ms").dt.date
    clean_fitness_df["date"] = pd.to_datetime(clean_fitness_df["date"])
    clean_fitness_df["beginTimestamp"] = pd.to_datetime(clean_fitness_df["beginTimestamp"], unit="ms")

    print("✅ Fitness cleaned")
    return clean_fitness_df

############################ MERGING FUNCTIONS #################################

def merge_all_data() -> pd.DataFrame:

    sleep_df = clean_wellness()
    fitness_df = clean_fitness()
    aggregator  = clean_data_aggregator()

    sleep_fitness = pd.merge(sleep_df, fitness_df, on="date", how="left")
    sleep_fitness = sleep_fitness.rename(columns={'date': 'calendarDate'})
    sleep_fitness = sleep_fitness.sort_values(by='calendarDate', ascending=True)
    sleep_fitness.set_index('calendarDate',inplace=True)

    final_dataset = pd.merge(sleep_fitness, aggregator,left_index=True, right_index=True, how='left')

    return final_dataset

def load_to_csv(output_path="raw_data/"):
    complet_path = os.path.join(output_path, "fitness_data.csv")
    final_df = merge_all_data()
    final_df.to_csv(complet_path,index=True)
    print("✅ Save as csv")

load_to_csv()
