import pyarrow.parquet as pq
import pandas
import pandas as pd
import os

class ResultsManager(object):

    def __init__(self):
        self.results_cache = {}
        self.results_filepaths = {}

    def load_results(self, leaderboard_name, result_filepath, default_columns):
        if leaderboard_name in self.results_cache:
            return self.results_cache[leaderboard_name]

        self.results_filepaths[leaderboard_name] = result_filepath

        if os.path.exists(result_filepath):
            # pq.read_table(result_filepath)
            df = pd.read_parquet(result_filepath)
        else:
            df = pd.DataFrame(columns=default_columns)

        self.results_cache[leaderboard_name] = df

        return df

    def update_results(self, leaderboard_name, df):
        self.results_cache[leaderboard_name] = df

    def save(self, leaderboard_name: str):
        if leaderboard_name in self.results_filepaths:
            filepath = self.results_filepaths[leaderboard_name]

            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

            df = self.results_cache[leaderboard_name]
            df.to_parquet(filepath)

    def save_all(self):
        for leaderboard_name in self.results_filepaths.keys():
            self.save(leaderboard_name)

    def filter_primary_key(self, df: pandas.DataFrame, submission_epoch_str: str, data_split_name: str, actor_uuid: str):
        check_submission_timestamp = df['submission_timestamp'] == submission_epoch_str
        check_data_split = df['data_split'] == data_split_name
        check_actor_uuid = df['actor_UUID'] == actor_uuid

        unique_entry = check_submission_timestamp & check_data_split & check_actor_uuid

        if unique_entry.any():
            return df[unique_entry]
        else:
            return None

