# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

from leaderboards import metrics
from data_science import utils


def main(global_results_csv_filepath, output_dirpath):
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # load the data
    results_df = pd.read_csv(global_results_csv_filepath)
    results_df.fillna(value=np.nan, inplace=True)
    results_df.replace(to_replace=[None], value=np.nan, inplace=True)
    results_df.replace(to_replace='None', value=np.nan, inplace=True)

    results_df = utils.filter_dataframe_by_cross_entropy_threshold(results_df, (0.3465 + 0.1))

    # get the unique set of teams
    teams = set(results_df['team_name'].to_list())

    fig = plt.figure(figsize=(6, 4.5), dpi=300)

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

            elementwise_ce = metrics.elementwise_binary_cross_entropy(predictions, targets)

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

