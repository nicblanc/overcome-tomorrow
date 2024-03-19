import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def load_all_file_in_directory(path, is_dict=False, type_file=".json") -> pd.DataFrame:
    """
    load all file in directory and create a dataframe
    type_file by default ".json"
    is_dict by default is False (if file return a dataframe of dict)
    """
    files = os.listdir(path)
    list_data = []
    for file in files:
        if file.endswith(type_file):
            data = pd.read_json(f"{path}{file}")
            if not(is_dict):
                list_data.append(data)
            else:
                list_data.append(pd.DataFrame(data["summarizedActivitiesExport"][0]))
    df = pd.concat(list_data, ignore_index=True)
    return df

# DONT FORGET TO CHANGE PATH 

def clean_sleep() -> pd.DataFrame:
    sleep_df = load_all_file_in_directory("../raw_data/Wellness/")
    
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
    return clean_sleep_df_copy
    
    
# DONT FORGET PATH IF CHANGE

def clean_fitness() -> pd.DataFrame:
    fitness_df = load_all_file_in_directory("../raw_data/Fitness/", is_dict=True)
    
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
    
    # CONVERT DATETIME 
    
    clean_fitness_df["date"] = pd.to_datetime(clean_fitness_df["beginTimestamp"], unit="ms").dt.date
    clean_fitness_df["date"] = pd.to_datetime(clean_fitness_df["date"])
    clean_fitness_df["beginTimestamp"] = pd.to_datetime(clean_fitness_df["beginTimestamp"], unit="ms")
    
    return clean_fitness_df


def merge_sleep_fitness() -> pd.DataFrame:
    
    sleep_df = clean_sleep()
    fitness_df = clean_fitness()
    
    merged_df = pd.merge(sleep_df, fitness_df, on="date", how="inner") 
    return merged_df