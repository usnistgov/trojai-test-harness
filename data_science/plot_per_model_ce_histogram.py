import os
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

from actor_executor import ground_truth
from data_science import utils


def main(global_results_csv_filepath, output_dirpath):
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # load the data
    results_df = pd.read_csv(global_results_csv_filepath)
    results_df.fillna(value=np.nan, inplace=True)
    results_df.replace(to_replace=[None], value=np.nan, inplace=True)
    results_df.replace(to_replace='None', value=np.nan, inplace=True)

    results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, 0.5)

    # get the unique set of teams
    teams = set(results_df['team_name'].to_list())

    fig = plt.figure(figsize=(5, 4), dpi=100)

    for team in teams:
        # get sub dataframe for this team
        team_df = results_df[results_df['team_name'] == team]

        # get the set of execution time stamps for this team
        timestamps = set(team_df['execution_time_stamp'].to_list())

        for timestamp in timestamps:
            # get the dataframe for this execution
            run_df = team_df[team_df['execution_time_stamp'] == timestamp]

            predictions = run_df['predicted'].to_numpy().reshape(-1)
            targets = run_df['ground_truth'].to_numpy().reshape(-1)

            predictions[np.isnan(predictions)] = 0.5

            elementwise_ce = ground_truth.binary_cross_entropy(predictions, targets)

            bins = np.arange(0, 10, 0.1).tolist()
            plt.hist(elementwise_ce, bins=100)
            plt.title('Per Model CE Loss Histogram')
            plt.savefig(os.path.join(output_dirpath, '{}-{}.png'.format(team, timestamp)))
            plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to generate a histogram of per-model CE loss for TrojAI challenge participants.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True)
    parser.add_argument('--output-dirpath', type=str, required=True, help='Where to save the output results')

    args = parser.parse_args()
    main(args.global_results_csv_filepath, args.output_dirpath)

