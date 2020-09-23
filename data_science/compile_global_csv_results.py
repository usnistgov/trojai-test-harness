import os
import numpy as np
import pandas as pd
import actor_executor.ground_truth


def find_dirs(fp):
    # find all directories in the results folder
    dirs = [d for d in os.listdir(fp) if os.path.isdir(os.path.join(fp, d))]
    return dirs


def main(test_harness_dirpath, server, metadata_filepath, output_dirpath):
    if server not in ['sts','es']:
        raise RuntimeError('{} is an invalid server option, should be "sts" or "es"'.format(server))
    if not os.path.exists(metadata_filepath):
        raise RuntimeError('metadata_filepath = {} does not exist.'.format(metadata_filepath))

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    results_fp = os.path.join(test_harness_dirpath, server, 'results')
    metadata = pd.read_csv(metadata_filepath)
    metadata.fillna(value=np.nan, inplace=True)
    metadata.replace(to_replace=[None], value=np.nan, inplace=True)
    metadata.replace(to_replace='None', value=np.nan, inplace=True)

    columns = list(metadata.columns)

    # Prep the output csv file with the headers
    global_results_csv_fp = os.path.join(output_dirpath, '{}-global-results.csv'.format(server))
    with open(global_results_csv_fp, 'w') as fh:
        fh.write('team_name,execution_time_stamp,ground_truth,predicted,cross_entropy')
        for col_name in columns:
            fh.write(',{}'.format(col_name))
        fh.write('\n')

        # find all team directories in the results folder
        teams = find_dirs(results_fp)
        teams.sort()

        for team in teams:
            team_fp = os.path.join(results_fp, team)
            # find all executions
            runs = find_dirs(team_fp)
            runs.sort()

            for run in runs:
                run_fp = os.path.join(team_fp, run)

                models = [f.replace('.txt','') for f in os.listdir(run_fp) if f.startswith('id-')]
                models.sort()

                for model in models:
                    row = metadata[metadata["model_name"] == model]
                    if len(row) == 0:
                        raise RuntimeError('No metadata for model {}'.format(model))

                    predicted = np.asarray(np.nan)
                    predicted_fp = os.path.join(run_fp, model + '.txt')
                    if os.path.exists(predicted_fp):
                        try:
                            with open(predicted_fp, 'r') as model_fh:
                                predicted = np.asarray(model_fh.read(), dtype=np.float64)
                        except:
                            predicted = np.asarray(np.nan)
                    target = row['poisoned'].to_numpy(dtype=np.float64)[0]
                    elementwise_ce = actor_executor.ground_truth.binary_cross_entropy(predicted, target)
                    ce = float(np.mean(elementwise_ce))

                    # fh.write('TeamName,ExecutionTimeStamp,ExecutionTimeStr,ModelId,GroundTruth,Predicted\n')
                    fh.write('{},{},{},{},{}'.format(team, run, target, predicted, ce))
                    for col_name in columns:
                        val = row[col_name].to_numpy()[0]
                        fh.write(',{}'.format(val))
                    fh.write('\n')


if __name__ == "__main__":
    import argparse

    # TODO this script will need to be modified for each round

    parser = argparse.ArgumentParser(description='Script which converts the TrojAi test harness data folder structure into a csv file of per model data.')
    parser.add_argument('--test-harness-dirpath', type=str, required=True)
    parser.add_argument('--server', type=str, required=True)
    parser.add_argument('--metadata-filepath', type=str, help="This is the metadata file released with the dataset being the detector is being evaluated against on the test server. I.e. the test dataset METADATA.csv file", required=True)
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()
    test_harness_dirpath = args.test_harness_dirpath
    server = args.server.lower()
    metadata_filepath = args.metadata_filepath
    output_dirpath = args.output_dirpath

    print('test_harness_dirpath = {}'.format(test_harness_dirpath))
    print('server = {}'.format(server))
    print('metadata_filepath = {}'.format(metadata_filepath))
    print('output_dirpath = {}'.format(output_dirpath))

    main(test_harness_dirpath, server, metadata_filepath, output_dirpath)

