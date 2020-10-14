# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import pandas as pd
import sklearn.metrics

from actor_executor import metrics


def find_dirs(fp):
    # find all directories in the results folder
    dirs = [d for d in os.listdir(fp) if os.path.isdir(os.path.join(fp, d))]
    return dirs


def main(global_results_csv_filepath, queue, output_dirpath):
    # load the global results csv into a data frame
    results_df = pd.read_csv(global_results_csv_filepath)

    # create output file
    with open(os.path.join(output_dirpath, '{}-leaderboard-summary.csv'.format(queue)), 'w') as fh:
        fh.write('Team, ExecutionTimeStamp, CrossEntropyLoss, CrossEntropy95ConfidenceInterval, ROC-AUC\n')

        # get the unique set of teams
        teams = set(results_df['team_name'].to_list())

        for team in teams:
            # get sub dataframe for this team
            team_df = results_df[results_df['team_name'] == team]

            # get the set of execution time stamps for this team
            timestamps = set(team_df['execution_time_stamp'].to_list())

            for timestamp in timestamps:
                # get the dataframe for this execution
                run_df = team_df[team_df['execution_time_stamp'] == timestamp]

                targets = run_df['ground_truth'].to_numpy(dtype=np.float32)
                predictions = run_df['predicted'].to_numpy(dtype=np.float32)
                ce = run_df['cross_entropy'].to_numpy(dtype=np.float32)
                ci = metrics.cross_entropy_confidence_interval(ce)
                ce = np.mean(ce)

                TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = metrics.confusion_matrix(targets, predictions)
                roc_auc = sklearn.metrics.auc(FPR, TPR)

                fh.write('{}, {}, {}, {}, {}\n'.format(team, timestamp, ce, ci, roc_auc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script build the global results csv filepath.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True)
    parser.add_argument('--queue', type=str, required=True)
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()

    main(args.global_results_csv_filepath, args.queue, args.output_dirpath)



