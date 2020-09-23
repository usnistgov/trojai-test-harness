import os
import numpy as np
import pandas as pd
import sklearn.metrics

from actor_executor import ground_truth


def find_dirs(fp):
    # find all directories in the results folder
    dirs = [d for d in os.listdir(fp) if os.path.isdir(os.path.join(fp, d))]
    return dirs


def main(global_results_csv_filepath, queue, output_dirpath):
    # load the global results csv into a data frame
    results_df = pd.read_csv(global_results_csv_filepath)

    # create output file
    with open(os.path.join(output_dirpath, '{}-leaderboard-summary.csv'.format(queue)), 'w') as fh:
        fh.write('Team, CrossEntropyLoss, CrossEntropy95ConfidenceInterval, ROC-AUC, ExecutionTimeStamp\n')

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
                ci = ground_truth.cross_entropy_confidence_interval(ce)
                ce = np.mean(ce)

                TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = ground_truth.gen_confusion_matrix(targets, predictions)
                roc_auc = sklearn.metrics.auc(FPR, TPR)

                fh.write('{}, {}, {}, {}, {}\n'.format(team, ce, ci, roc_auc, timestamp))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script build the global results csv filepath.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True)
    parser.add_argument('--queue', type=str, required=True)
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()

    main(args.global_results_csv_filepath, args.queue, args.output_dirpath)



