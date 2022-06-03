import pandas as pd
import configargparse
import json
from collections import OrderedDict
import numpy as np
import os

class JSONConfigFileParser(configargparse.ConfigFileParser):
    def get_syntax_description(self):
        return ["Config file syntax alled based on JSON format"]

    def parse(self, stream):
        try:
            parsed_obj = json.load(stream)
        except Exception as e:
            raise(configargparse.ConfigFileParserException("Couldn't parse config file: %s" % e))

        result = OrderedDict()
        for key, value in parsed_obj.items():
            if isinstance(value, list):
                result[key] = value
            elif value is None:
                pass
            else:
                result[key] = str(value)

        return result

    def serialize(self, items):
        items = dict(items)
        return json.dumps(items, indent=2, sort_keys=True)

def elementwise_binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray, epsilon=1e-12) -> np.ndarray:
    predictions = predictions.astype(np.float64)
    targets = targets.astype(np.float64)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    a = targets * np.log(predictions)
    b = (1 - targets) * np.log(1 - predictions)
    ce = -(a + b)
    return ce

def parse_dataset_mapping(values):
    result = {}
    if values is not None:
        for item in values:
            item_split = item.split('=')
            key = item_split[0].strip()
            if len(item_split) > 1:
                value = '='.join(item_split[1:])

            result[key] = value
    return result

def build_round_dataset_csv(round_dataset_dirpath, output_dirpath, overwrite_csv=True, metadata_filename='METADATA.csv', ground_truth_filename='ground_truth.csv', skip_leftovers=True):

    round_name = os.path.basename(round_dataset_dirpath)
    output_filepath = os.path.join(output_dirpath, '{}_METADATA.csv'.format(round_name))

    if os.path.exists(output_filepath) and not overwrite_csv:
        print('Skipping building round metadata: {} already exists and overwrite is disabled.'.format(output_filepath))
        return pd.read_csv(output_filepath)

    all_df_list = []

    for dir in os.listdir(round_dataset_dirpath):

        round_dirpath = os.path.join(round_dataset_dirpath, dir)

        if skip_leftovers:
            if 'leftover' in dir:
                print('Skipping leftovers: {}'.format(round_dirpath))
                continue

        metadata_filepath = os.path.join(round_dirpath, metadata_filename)

        if not os.path.exists(metadata_filepath):
            print('Skipping {}, it does not contain the metadata file: {}'.format(dir, metadata_filepath))
            continue

        dataset_name_split = dir.split('-')
        if len(dataset_name_split) == 3:
            data_split_name = dataset_name_split[1]
        else:
            data_split_name = dir

        df = pd.read_csv(metadata_filepath)

        # Add column for data_split
        new_df = df.assign(data_split=data_split_name)

        # Add column for ground_truth
        new_df = new_df.assign(ground_truth='NaN')

        models_dir = os.path.join(round_dirpath, 'models')

        if not os.path.exists(models_dir):
            print('{} does not have a models dir, attempting to find ids in {}'.format(models_dir, round_dirpath))

            model_dir = os.path.join(round_dirpath, 'id-00000001')
            if os.path.exists(model_dir):
                models_dir = round_dirpath
                print('Found model id in {}, using {} as model dir'.format(model_dir, models_dir))
            else:
                print('Failed to find model ids for: {}'.format(model_dir))
                models_dir = None

        if models_dir is not None:
            # Add ground truth values into data
            for model_name in os.listdir(models_dir):
                model_dirpath = os.path.join(models_dir, model_name)

                # Skip model_name that is not a directory
                if not os.path.isdir(model_dirpath):
                    continue

                ground_truth_filepath = os.path.join(models_dir, model_name, ground_truth_filename)
                if not os.path.exists(ground_truth_filepath):
                    print('WARNING, ground truth file does not exist: {}'.format(ground_truth_filepath))
                    continue

                with open(ground_truth_filepath, 'r') as f:
                    data = float(f.read())
                    new_df.loc[new_df['model_name'] == model_name, 'ground_truth'] = data

        all_df_list.append(new_df)

    all_df = pd.concat(all_df_list)

    # Rearrange columns slightly
    columns = list(all_df.columns.values)
    column_order = ['model_name', 'ground_truth', 'data_split']
    remove_columns = ['converged', 'nonconverged_reason']

    # Remove columns
    for column_name in remove_columns:
        if column_name in columns:
            columns.remove(column_name)

    # Reorder columns
    index = 0
    for column_name in column_order:
        if column_name in columns:
            columns.remove(column_name)
            columns.insert(index, column_name)
            index += 1

    all_df = all_df[columns]

    all_df.to_csv(output_filepath, index=False)
    print('Finished writing round metadata to {}'.format(output_filepath))

    return all_df

def build_round_results(df, round_results_dirpath, output_dirpath, result_dataset_mapping, overwrite_csv=True):
    round_name = os.path.basename(round_results_dirpath)
    output_filepath = os.path.join(output_dirpath, '{}_RESULTS.csv'.format(round_name))

    if os.path.exists(output_filepath) and not overwrite_csv:
        print('Skipping building round results: {} already exists and overwrite is disabled.'.format(output_filepath))
        return pd.read_csv(output_filepath)

    all_dfs = []

    for result_name, data_split_str in result_dataset_mapping.items():
        round_result_dirpath = os.path.join(round_results_dirpath, result_name)
        teams_round_results_dirpath = os.path.join(round_result_dirpath, 'results')

        if not os.path.exists(teams_round_results_dirpath):
            print('Failed to find round results directory: {}'.format(teams_round_results_dirpath))
            continue

        for team_name in os.listdir(teams_round_results_dirpath):
            team_result_dirpath = os.path.join(teams_round_results_dirpath, team_name)
            for timestamp in os.listdir(team_result_dirpath):
                team_timestamp_result_dirpath = os.path.join(team_result_dirpath, timestamp)

                team_results = {}


                for result_file in os.listdir(team_timestamp_result_dirpath):
                    if result_file.startswith('id-') and result_file.endswith('.txt'):
                        with open(os.path.join(team_timestamp_result_dirpath, result_file), 'r') as f:
                            data = f.read()
                            model_id = result_file.split('.')[0]
                            if data == '':
                                data = float('nan')
                            try:
                                team_results[model_id] = float(data)
                            except ValueError:
                                team_results[model_id] = float('nan')

                # Gather corresponding results
                all_model_ids = list(df.loc[df['data_split'] == data_split_str, 'model_name'].unique())
                predictions = []
                targets = []
                for model_id in all_model_ids:
                    if model_id in team_results:
                        predictions.append(team_results[model_id])
                    else:
                        predictions.append(np.nan)

                    target_value = df.loc[(df['data_split'] == data_split_str) & (df['model_name'] == model_id), 'ground_truth'].item()
                    targets.append(target_value)

                cross_entropy = elementwise_binary_cross_entropy(np.array(predictions), np.array(targets))

                new_data = {}
                new_data['model_name'] = all_model_ids
                new_data['team_name'] = [team_name] * len(all_model_ids)
                new_data['submission_date'] = [timestamp] * len(all_model_ids)
                new_data['data_split'] = [data_split_str] * len(all_model_ids)
                new_data['prediction'] = predictions
                new_data['ground_truth'] = targets
                new_data['cross_entropy'] = [float(i) for i in cross_entropy]

                new_df = pd.DataFrame(new_data)
                all_dfs.append(new_df)

    result_df = pd.concat(all_dfs)

    result_df.to_csv(output_filepath, index=False)

    print('Finished writing round results to {}'.format(output_filepath))

    return result_df

def build_round_csvs(round_dataset_dirpath, round_results_dirpath, output_dirpath, result_dataset_mapping, overwrite_csv=True):

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # Build the round CSV
    round_df = build_round_dataset_csv(round_dataset_dirpath, output_dirpath, overwrite_csv=overwrite_csv)

    # Build round results
    build_round_results(round_df, round_results_dirpath, output_dirpath, result_dataset_mapping, overwrite_csv=overwrite_csv)



if __name__ == '__main__':

    parser = configargparse.ArgumentParser(
        config_file_parser_class=JSONConfigFileParser,
        description='Builds the CSV files related to a round. Two CSVs are output, METADATA CSV describing the training and a RESULTS CSV describing the results of the round'
    )
    parser.add_argument('--config-filepath', is_config_file=True, help='The filepath to the config file.')

    parser.add_argument('--save-config-filepath', type=str, help='The path to save the config file.')
    parser.add_argument('--round-dataset-dirpath', type=str, help='The dirpath to the round datasets.')
    parser.add_argument('--round-results-dirpath', type=str, help='The dirpath to the round results.')
    parser.add_argument('--output-dirpath', type=str, help='Output dirpath for CSVs.')
    parser.add_argument('--result-dataset-mapping', metavar="KEY=VALUE", nargs='+', help='Selects which directories to use when processing results. '
                                                                                         'Creates the key-value pair mapping between the result directory folder and round dataset folder. '
                                                                                         'This is used to idenfity the ground truth to be used when computing cross entropy. If a value requires a space, then should be'
                                                                                         'defined in double quotes, such as: "round result dir"="round dataset dir"')

    args = parser.parse_args()

    if args.save_config_filepath is not None:
        parser.write_config_file(args, [args.save_config_filepath])

    build_round_csvs(args.round_dataset_dirpath, args.round_results_dirpath, args.output_dirpath, parse_dataset_mapping(args.result_dataset_mapping))