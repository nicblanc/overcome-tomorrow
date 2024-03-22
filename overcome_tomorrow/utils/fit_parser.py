from garmin_fit_sdk import Decoder, Stream, Profile
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
import pandas as pd
import numpy as np

FILTERED_SESSION_KEYS = {
    178,  # est sweat loss
    205,  # beginning potential stamina
    206,  # ending potential stamina
    207,  # min stamina
    188,  # Primary benefit
    'avg_cadence',
    'avg_heart_rate',
    'avg_power',
    'avg_step_length',
    'avg_stroke_distance',
    'enhanced_avg_speed',
    'enhanced_max_speed',
    'max_cadence',
    'max_heart_rate',
    'max_power',
    'max_running_cadence',
    'normalized_power',
    'num_active_lengths',
    'num_lengths',
    'pool_length',
    'pool_length_unit',
    'sport',
    'start_time',
    'sub_sport',
    'timestamp',
    'total_anaerobic_training_effect',
    'total_ascent',
    'total_calories',
    'total_descent',
    'total_distance',
    'total_strokes',
    'total_training_effect',
    'training_load_peak',
    'training_stress_score'
}

FILTERED_LENGTH_KEYS = {
    'avg_speed',
    'avg_swimming_cadence',
    'start_time',
    'timestamp',
    'total_strokes'
}

FILTERED_SET_KEYS = {
    'category',
    'duration',
    'repetitions',
    'set_type',
    'start_time',
    'timestamp',
    'weight',
    'weight_display_unit'
}

FILTERED_RECORD_KEYS = {
    137,  # Stamina
    138,  # Potential Stamina
    90,  # Perf Cond
    'accumulated_power',
    'cadence',
    'distance',
    'enhanced_speed',
    'heart_rate',
    'power',
    'timestamp'
}


def init_filtered_extracted_dataset():
    """ Init dict to extract data from FIT file using callback """
    record = {}
    for key in FILTERED_RECORD_KEYS:
        record[key] = []

    length = {}
    for key in FILTERED_LENGTH_KEYS:
        length[key] = []

    sets = {}
    for key in FILTERED_SET_KEYS:
        sets[key] = []

    session = {}
    for key in FILTERED_SESSION_KEYS:
        session[key] = []
    return {
        "SESSION": session,
        "RECORD": record,
        "SET": sets,
        "LENGTH": length
    }


def fit_file_to_dataframe(file, extractor):
    """ Decode fit file with an exctractor callback """
    stream = Stream.from_file(file)
    decoder = Decoder(stream)
    decoder.read(mesg_listener=extractor)


def init_extractor():
    """ Init exctractor callback that will be used durint FIT file decoding"""
    filtered_messages_dict = init_filtered_extracted_dataset()

    def filtered_extract_listener(mesg_num, message):
        for message_name, message_dict in filtered_messages_dict.items():
            if mesg_num == Profile['mesg_num'][message_name]:
                for key in message_dict.keys():
                    if key in message:
                        message_dict[key].append(message[key])
                    else:
                        message_dict[key].append(np.nan)

    return filtered_messages_dict, filtered_extract_listener


def process_fit_files(input_path, output_path):
    """ Process all files given an input path. An activity file will be created in the output path, together with 2 sub_directories containing details activities fields """
    output_path_full_activity = join(output_path, "FullRecords/")
    output_path_activity = join(output_path, "Records/")
    # if the output directories are not present
    # then create them.
    if not exists(output_path_full_activity):
        makedirs(output_path_full_activity)
    if not exists(output_path_activity):
        makedirs(output_path_activity)

    all_sessions = {}
    for key in FILTERED_SESSION_KEYS:
        all_sessions[key] = []
    all_sessions["activity_id"] = []
    all_sessions_df = pd.DataFrame(all_sessions).set_index("timestamp")

    for f in tqdm(listdir(input_path)):
        file_path = join(input_path, f)
        if isfile(file_path):
            dataset, extractor = init_extractor()
            activity_id = f.split(".fit")[0]

            path_output_file = join(output_path_activity, f"{activity_id}.csv")
            path_output_file_full = join(
                output_path_full_activity, f"{activity_id}_full.csv")

            fit_file_to_dataframe(file_path, extractor=extractor)
            dataset_df = pd.DataFrame.from_dict(
                dataset["SESSION"]).set_index("timestamp")
            dataset_df["activity_id"] = activity_id
            record_df = pd.DataFrame.from_dict(
                dataset["RECORD"]).set_index("timestamp")
            activity_df = pd.concat([record_df,
                                     pd.DataFrame.from_dict(
                                         dataset["LENGTH"]).set_index("timestamp"),
                                     pd.DataFrame.from_dict(dataset["SET"]).set_index("timestamp")])
            record_df.to_csv(path_output_file)
            activity_df.to_csv(path_output_file_full)

            all_sessions_df = pd.concat([all_sessions_df, dataset_df])

    all_sessions_df.to_csv(join(output_path, "activities.csv"))
