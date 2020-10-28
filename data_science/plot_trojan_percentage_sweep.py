# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import pandas as pd
import random
import sklearn.metrics

from collections import OrderedDict
from matplotlib import pyplot as plt

from actor_executor import metrics
from data_science import utils


def plot_sweep(df, team_name, timestamp, output_fp, nb_reps):
    # ensure we have only a single run in the data frame
    # subset to team name
    df = df[df["team_name"] == team_name]
    # subset to timestamp
    df = df[df["execution_time_stamp"] == timestamp]

    # get list of the model ids that exists within the run
    models = df['model_name'].to_list()

    # setup data structures
    ground_truth_dict = OrderedDict()
    results = OrderedDict()
    results_guess = OrderedDict()
    poisoned_model_dict = dict()
    clean_model_dict = dict()

    # loop over the model ids and extract the predictions and targets for each model id
    for model in models:
        row = df[df["model_name"] == model]
        results_guess[model] = np.nan
        predicted = row['predicted'].to_numpy(dtype=np.float32)[0]
        results[model] = predicted
        target = row['ground_truth'].to_numpy(dtype=np.float32)[0]
        ground_truth_dict[model] = target

        # populate dictionaries based on whether the model was poisoned or not
        if target > 0:
            poisoned_model_dict[model] = predicted
        else:
            clean_model_dict[model] = predicted

    trojan_percentage_list = list()
    ce_loss_list = list()
    roc_auc_list = list()
    ce_loss_guess_list = list()

    # loop over the trojan percentages to build a subset of the models for each trojan percentage
    for tgt_trojan_percentage in range(1, 99, 1):
        # obtain the trojan percentage as a number in [0, 1]
        tgt_trojan_percentage = tgt_trojan_percentage / 100.0

        # for stability, build N copies of this trojan percentage
        for n in range(nb_reps):
            clean_keys = list(clean_model_dict.keys())
            random.shuffle(clean_keys)
            poisoned_keys = list(poisoned_model_dict.keys())
            random.shuffle(poisoned_keys)

            nb_poisoned = 0
            subset_dict = dict()

            while True:
                if len(subset_dict) == 0:
                    current_trojan_percentage = 0
                else:
                    current_trojan_percentage = nb_poisoned / len(subset_dict)
                if current_trojan_percentage < tgt_trojan_percentage:
                    if len(poisoned_keys) == 0:
                        # we have run out of keys, the subset dataset is as complete as we can mak it
                        break
                    nb_poisoned = nb_poisoned + 1
                    k = poisoned_keys.pop(0)
                    subset_dict[k] = poisoned_model_dict[k]
                else:
                    if len(clean_keys) == 0:
                        # we have run out of keys, the subset dataset is as complete as we can mak it
                        break
                    k = clean_keys.pop(0)
                    subset_dict[k] = clean_model_dict[k]

            predictions = list()
            predictions_guess = list()
            targets = list()

            for k in subset_dict.keys():
                t = ground_truth_dict[k]
                targets.append(t)
                p = results[k]
                predictions.append(p)
                g = results_guess[k]
                predictions_guess.append(g)

            predictions = np.array(predictions).reshape(-1, 1)
            predictions_guess = np.array(predictions_guess).reshape(-1, 1)
            targets = np.array(targets).reshape(-1, 1)
            # replace nans (missing) with the base rate
            predictions[np.isnan(predictions)] = tgt_trojan_percentage
            predictions_guess[np.isnan(predictions_guess)] = tgt_trojan_percentage

            elementwise_ce = metrics.elementwise_binary_cross_entropy(predictions, targets)
            ce = np.mean(elementwise_ce)
            elementwise_ce_guess = metrics.elementwise_binary_cross_entropy(predictions_guess, targets)
            ce_guess = np.mean(elementwise_ce_guess)

            TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = metrics.confusion_matrix(targets, predictions)
            roc_auc = sklearn.metrics.auc(FPR, TPR)

            trojan_percentage_list.append(tgt_trojan_percentage)
            ce_loss_list.append(ce)
            roc_auc_list.append(roc_auc)
            ce_loss_guess_list.append(ce_guess)

    if not os.path.exists(output_fp):
        os.makedirs(output_fp)
    if not os.path.exists(os.path.join(output_fp, 'ce-sweep')):
        os.makedirs(os.path.join(output_fp, 'ce-sweep'))
    if not os.path.exists(os.path.join(output_fp, 'roc-auc-sweep')):
        os.makedirs(os.path.join(output_fp, 'roc-auc-sweep'))

    fig = plt.figure(figsize=(16, 9), dpi=100)
    x = np.asarray(trojan_percentage_list)

    y1 = np.asarray(ce_loss_list)
    y2 = np.asarray(ce_loss_guess_list)
    y3 = y2 / 2
    ax = plt.gca()
    plt.cla()
    h1 = ax.scatter(x, y1)
    h2 = ax.scatter(x, y2)
    h3 = ax.scatter(x, y3)
    plt.title('CE-Loss as a function of Poisoning Percentage for {}-{}'.format(team_name, timestamp))
    plt.xlabel('Trojan Percentage')
    plt.ylabel('CE-Loss')
    plt.legend(handles=[h1, h2, h3], labels=['CE-Loss', 'Guessing-BaseRate', 'Success'])
    plt.savefig(os.path.join(output_fp, 'ce-sweep', '{}-{}-ce-trojan-sweep.png'.format(team_name, timestamp)))

    y = np.asarray(roc_auc_list)
    ax = plt.gca()
    plt.cla()
    ax.scatter(x, y)
    plt.title('ROC-AUC as a function of Poisoning Percentage for {}-{}'.format(team_name, timestamp))
    plt.xlabel('Trojan Percentage')
    plt.ylabel('ROC-AUC')
    plt.savefig(os.path.join(output_fp, 'roc-auc-sweep', '{}-{}-roc-auc-sweep.png'.format(team_name, timestamp)))
    plt.close(fig)


def main(global_results_csv_filepath, nb_reps, output_dirpath):

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

            plot_sweep(run_df, team, timestamp, output_dirpath, nb_reps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to evaluate how well a solution detections trojans when trojans are rarer by subsetting the full held out test dataset.')
    parser.add_argument('--global-results-csv-filepath', type=str, required=True,
                        help='The csv filepath holding the global results data.')

    parser.add_argument('--nb-reps', type=int, help='Number of times to generate the subset.', default=10)
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()
    print('global_results_csv_filepath = {}'.format(args.global_results_csv_filepath))
    print('nb_reps = {}'.format(args.nb_reps))
    print('output_dirpath = {}'.format(args.output_dirpath))
    main(args.global_results_csv_filepath, args.nb_reps, args.output_dirpath)


