import numpy as np
import pandas as pd
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from leaderboards import time_utils


class SummaryMetric(object):
    def __init__(self, share_with_collaborators: bool, add_to_html: bool):
        self.shared_with_collaborators = share_with_collaborators
        self.add_to_html = add_to_html

    def compute_and_write_data(self, leaderboard_name: str, data_split_name: str,  metadata_df: pd.DataFrame, results_df: pd.DataFrame, output_dirpath: str):
        pass


class SummaryAverageCEOverTime(SummaryMetric):
    def __init__(self):
        super().__init__(share_with_collaborators=True, add_to_html=True)

    def compute_and_write_data(self, leaderboard_name: str, data_split_name: str,  metadata_df: pd.DataFrame, results_df: pd.DataFrame, output_dirpath: str):
        # Get list of unique timestamps from results
        unique_timestamps = results_df['submission_timestamp'].unique()

        if len(unique_timestamps) == 0:
            return []

        unique_timestamps.sort()
        avg_ce_scores = np.zeros(len(unique_timestamps))
        index = 0

        for timestamp in unique_timestamps:
            # Get cross entropy and average it
            actor_results_df = results_df[results_df['submission_timestamp'] == timestamp]
            actor_names = actor_results_df['team_name'].unique()
            if len(actor_names) != 1:
                logging.error('{} has multiple actor names: {}'.format(timestamp, actor_names))
                continue

            # TODO: Add a legend and color-code each actor
            actor_name = actor_names[0]

            ce_values = actor_results_df['cross_entropy'].values

            ce_score = np.average(ce_values).item()
            avg_ce_scores[index] = ce_score
            index += 1

        actor_dates = [time_utils.convert_to_datetime(ts) for ts in unique_timestamps]
        actor_dates = mdate.date2num(actor_dates)

        fig, ax = plt.subplots()
        ax.plot_date(actor_dates, avg_ce_scores)
        ax.set_title('Average Cross Entropy over time in {} for dataset {}'.format(leaderboard_name, data_split_name))
        ax.set_ylabel('Average Cross Entropy')
        ax.yaxis.grid(True)

        date_fmt = '%Y-%m-%dT%H:%M:%S'
        date_formatter = mdate.DateFormatter(date_fmt)
        ax.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()

        filepath = os.path.join(output_dirpath, 'avgce_{}_{}.png'.format(leaderboard_name, data_split_name))

        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        plt.clf()

        return [filepath]






