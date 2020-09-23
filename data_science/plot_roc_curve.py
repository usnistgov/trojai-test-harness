import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics

from actor_executor import ground_truth
from data_science import utils


def generate_roc_image(df, team_name, timestamp, output_fp):
    # ensure we have only a single run in the data frame
    # subset to team name
    df = df[df["team_name"] == team_name]
    # subset to timestamp
    df = df[df["execution_time_stamp"] == timestamp]

    if not os.path.exists(output_fp):
        os.makedirs(output_fp)
    if not os.path.exists(output_fp):
        os.makedirs(output_fp)

    targets = df['ground_truth']
    predictions = df['predicted']

    TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = ground_truth.gen_confusion_matrix(targets, predictions)
    roc_auc = sklearn.metrics.auc(FPR, TPR)

    fig = plt.figure(figsize=(5, 4), dpi=100)
    plt.clf()
    lw = 2
    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
    plt.plot(FPR, TPR, 'b-', marker='o', markersize=4, linewidth=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    legend_str = 'ROC AUC = {:02g}'.format(roc_auc)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend([legend_str], loc='lower right')
    plt.savefig(os.path.join(output_fp, '{}-{}-roc.png'.format(team_name, timestamp)))
    plt.close(fig)


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

    for team in teams:
        # get sub dataframe for this team
        team_df = results_df[results_df['team_name'] == team]

        # get the set of execution time stamps for this team
        timestamps = set(team_df['execution_time_stamp'].to_list())

        for timestamp in timestamps:
            # get the dataframe for this execution
            run_df = team_df[team_df['execution_time_stamp'] == timestamp]

            generate_roc_image(run_df, team, timestamp, output_dirpath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to plot a ROC curves.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')
    parser.add_argument('--output-dirpath', type=str, required=True, help='Where to save the output results')

    args = parser.parse_args()

    main(args.global_results_csv_filepath, args.output_dirpath)
