import pandas as pd
import os

def loading_data_aggregator():
    dossier = os.path.join('raw_data','Aggregator')
    dataframes=[]
    for fichier in os.listdir(dossier):
        complet_path = os.path.join(dossier, fichier)
        if fichier.endswith('.json'):
            df_test = pd.read_json(complet_path)
            dataframes.append(df_test)

    df = pd.concat(dataframes, ignore_index=True)
    df['calendarDate'] = pd.to_datetime(df['calendarDate'])
    df = df.sort_values(by='calendarDate', ascending=True)
    df = df.loc[df['calendarDate'] >= pd.Timestamp('2019-01-01')]
    df = df.drop_duplicates(subset=['calendarDate'], keep='first')
    df.set_index('calendarDate',inplace=True)
    print("✅ Aggregator loaded")
    return df

def clean_data_aggregator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean aggregator's dataset
        - assigning correct dtypes to each column
        - removing useless columns
        - deal with NaN
    """
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

    wellness = df.drop(columns=features_to_drop)
    print("✅ Aggregator cleaned")

    return wellness

def preprocessing():
    """
    Preprocessing the dataset
    """
    # CODE HERE
    pass
