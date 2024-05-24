# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import pandas as pd
import leaderboards.metrics


def find_dirs(fp):
    # find all directories in the results folder
    dirs = [d for d in os.listdir(fp) if os.path.isdir(os.path.join(fp, d))]
    return dirs


def main(results_dirpath, data_split, metadata_filepath, output_dirpath, round_name):
    if data_split not in ['train','test','holdout']:
        raise RuntimeError('{} is an invalid server option, should be "train" or "test" or "holdout"'.format(data_split))
    if not os.path.exists(metadata_filepath):
        raise RuntimeError('metadata_filepath = {} does not exist.'.format(metadata_filepath))

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    results_fp = os.path.join(results_dirpath, "{}-dataset".format(round_name), "{}-dataset".format(data_split), "{}-submission".format(round_name))
    metadata = pd.read_csv(metadata_filepath)
    metadata.fillna(value=np.nan, inplace=True)
    metadata.replace(to_replace=[None], value=np.nan, inplace=True)
    metadata.replace(to_replace='None', value=np.nan, inplace=True)

    models = metadata['model_name'].to_list()
    models.sort()

    columns = list(metadata.columns)

    ce_metric = leaderboards.metrics.AverageCrossEntropy(write_html=False, share_with_actor=False, store_result=False)

    df_list = list()
    meta_list = list()

    # find all team directories in the results folder
    teams = find_dirs(results_fp)
    teams.sort()

    for team in teams:
        team_fp = os.path.join(results_fp, team)
        # find all executions
        submissions = find_dirs(team_fp)
        submissions.sort()

        for submission in submissions:
            submission_fp = os.path.join(team_fp, submission)
            # find all executions
            executions = find_dirs(submission_fp)
            executions.sort()

            for execution in executions:
                execution_fp = os.path.join(submission_fp, execution)

                for model in models:
                    row = metadata[metadata["model_name"] == model]
                    if len(row) == 0:
                        raise RuntimeError('No metadata for model {}'.format(model))

                    predicted = np.asarray(np.nan)
                    predicted_fp = os.path.join(execution_fp, model + '.txt')
                    if os.path.exists(predicted_fp):
                        try:
                            with open(predicted_fp, 'r') as model_fh:
                                predicted = np.asarray(model_fh.read(), dtype=np.float64)
                        except:
                            predicted = np.asarray(np.nan)
                    target = row['poisoned'].to_numpy(dtype=np.float64)[0]

                    res = ce_metric.compute(predictions=predicted, targets=target)
                    # elementwise_ce = res['metadata']['cross_entropy']
                    ce = float(res['result'])

                    submission_timestamp = submission.split('-')[0]
                    execution_timestamp = execution.split('-')[0]

                    dat_dict = dict()
                    dat_dict['team_name'] = team
                    dat_dict['submission_time_stamp'] = submission_timestamp
                    dat_dict['execution_time_stamp'] = execution_timestamp
                    dat_dict['ground_truth'] = target
                    dat_dict['predicted'] = predicted
                    dat_dict['cross_entropy'] = ce

                    cd = pd.json_normalize(dat_dict)
                    df_list.append(cd)
                    meta_list.append(row)

    full_df = pd.concat(df_list, axis=0)
    meta_df = pd.concat(meta_list, axis=0)
    full_df = pd.concat([full_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
    global_results_csv_fp = os.path.join(output_dirpath, '{}-global-results.csv'.format(data_split))
    full_df.to_csv(global_results_csv_fp, index=False)


    # # Prep the output csv file with the headers
    # global_results_csv_fp = os.path.join(output_dirpath, '{}-global-results.csv'.format(data_split))
    # with open(global_results_csv_fp, 'w') as fh:
    #     fh.write('team_name,submission_time_stamp,execution_time_stamp,ground_truth,predicted,cross_entropy')
    #     for col_name in columns:
    #         fh.write(',{}'.format(col_name))
    #     fh.write('\n')
    #
    #     # find all team directories in the results folder
    #     teams = find_dirs(results_fp)
    #     teams.sort()
    #
    #     for team in teams:
    #         team_fp = os.path.join(results_fp, team)
    #         # find all executions
    #         submissions = find_dirs(team_fp)
    #         submissions.sort()
    #
    #         for submission in submissions:
    #             submission_fp = os.path.join(team_fp, submission)
    #             # find all executions
    #             executions = find_dirs(submission_fp)
    #             executions.sort()
    #
    #             for execution in executions:
    #                 execution_fp = os.path.join(submission_fp, execution)
    #
    #                 for model in models:
    #                     row = metadata[metadata["model_name"] == model]
    #                     if len(row) == 0:
    #                         raise RuntimeError('No metadata for model {}'.format(model))
    #
    #                     predicted = np.asarray(np.nan)
    #                     predicted_fp = os.path.join(execution_fp, model + '.txt')
    #                     if os.path.exists(predicted_fp):
    #                         try:
    #                             with open(predicted_fp, 'r') as model_fh:
    #                                 predicted = np.asarray(model_fh.read(), dtype=np.float64)
    #                         except:
    #                             predicted = np.asarray(np.nan)
    #                     target = row['poisoned'].to_numpy(dtype=np.float64)[0]
    #
    #                     res = ce_metric.compute(predictions=predicted, targets=target)
    #                     # elementwise_ce = res['metadata']['cross_entropy']
    #                     ce = float(res['result'])
    #
    #                     submission_timestamp = submission.split('-')[0]
    #                     execution_timestamp = execution.split('-')[0]
    #                     # fh.write('team_name,submission_time_stamp,execution_time_stamp,ground_truth,predicted,cross_entropy')
    #                     fh.write('{},{},{},{},{},{}'.format(team, submission_timestamp, execution_timestamp, target, predicted, ce))
    #                     for col_name in columns:
    #                         val = row[col_name].to_numpy()[0]
    #                         fh.write(',{}'.format(val))
    #                     fh.write('\n')


if __name__ == "__main__":
    import argparse

    # TODO this script will need to be modified for each round

    parser = argparse.ArgumentParser(description='Script which converts the TrojAi test harness data folder structure into a csv file of per model data.')
    parser.add_argument('--results_dirpath', type=str, required=True)
    parser.add_argument('--round_name', type=str, required=True)
    parser.add_argument('--data_split', type=str, required=True)
    parser.add_argument('--metadata_filepath', type=str, help="This is the metadata file released with the dataset being the detector is being evaluated against on the test server. I.e. the test dataset METADATA.csv file", required=True)
    parser.add_argument('--output_dirpath', type=str, required=True)

    args = parser.parse_args()
    results_dirpath = args.results_dirpath
    round_name = args.round_name
    data_split = args.data_split.lower()
    metadata_filepath = args.metadata_filepath
    output_dirpath = args.output_dirpath

    print('round_name = {}'.format(round_name))
    print('results_dirpath = {}'.format(results_dirpath))
    print('data_split = {}'.format(data_split))
    print('metadata_filepath = {}'.format(metadata_filepath))
    print('output_dirpath = {}'.format(output_dirpath))

    main(results_dirpath, data_split, metadata_filepath, output_dirpath, round_name)

