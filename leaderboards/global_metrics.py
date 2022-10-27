import pandas as pd

class GlobalMetric(object):
    def __init__(self):
        self.shared_with_collaborators = True
        self.additional_email_addresses = []

    def compute_and_write_data(self, leaderboard_name: str, data_split_name: str,  metadata_df: pd.DataFrame, results_df: pd.DataFrame, output_dirpath: str):
        pass


