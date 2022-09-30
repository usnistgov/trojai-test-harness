import pandas as pd
import configargparse
import json
from collections import OrderedDict
import numpy as np
import os

from leaderboards.trojai_config import TrojaiConfig
from leaderboards.submission import SubmissionManager, Submission
from leaderboards.leaderboard import Leaderboard
from leaderboards.actor import Actor, ActorManager
from leaderboards import time_utils

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


def build_round_dataset_csv(leaderboard: Leaderboard, output_dirpath, overwrite_csv=True, metadata_filename='METADATA.csv', ground_truth_filename='ground_truth.csv'):
    round_name = leaderboard.name
    output_filepath = os.path.join(output_dirpath, '{}_METADATA.csv'.format(round_name))

    all_df_list = []

    if os.path.exists(output_filepath) and not overwrite_csv:
        print('Skipping building round metadata: {} already exists and overwrite is disabled.'.format(output_filepath))
        return pd.read_csv(output_filepath)

    for split_name in leaderboard.get_all_data_split_names():
        dataset = leaderboard.get_dataset(split_name)
        dataset_dirpath = dataset.dataset_dirpath

        metadata_filepath = os.path.join(dataset_dirpath, metadata_filename)

        if not os.path.exists(metadata_filepath):
            print('Skipping {}, it does not contain the metadata file: {}'.format(dir, metadata_filepath))
            continue

        df = pd.read_csv(metadata_filepath)

        # Add column for data_split
        new_df = df.assign(data_split=split_name)

        # Add column for ground_truth
        new_df = new_df.assign(ground_truth='NaN')

        models_dir = os.path.join(dataset_dirpath, 'models')

        if os.path.exists(models_dir):
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
        else:
            print('{} does not exist'.format(models_dir))

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

def build_round_results(trojai_config: TrojaiConfig, leaderboard: Leaderboard, df: pd.DataFrame, output_dirpath, overwrite_csv=True):
    round_name = leaderboard.name
    output_filepath = os.path.join(output_dirpath, '{}_RESULTS.csv'.format(round_name))

    if os.path.exists(output_filepath) and not overwrite_csv:
        print('Skipping building round results: {} already exists and overwrite is disabled.'.format(output_filepath))
        return pd.read_csv(output_filepath)

    all_dfs = []

    submission_manager = SubmissionManager.load_json(leaderboard.submissions_filepath, leaderboard.name)
    actor_manager = ActorManager.load_json(trojai_config)
    default_result = leaderboard.get_default_prediction_result()

    for actor in actor_manager.get_actors():
        submissions = submission_manager.get_submissions_by_actor(actor)
        for data_split in leaderboard.get_all_data_split_names():
            leaderboard_metrics = leaderboard.get_submission_metrics(data_split)

            all_model_ids = list(df.loc[df['data_split'] == data_split, 'model_name'].unique())

            predictions = []
            targets = []
            metrics = {}
            for submission in submissions:
                if submission.data_split_name == data_split:

                    raw_predictions_np, raw_targets_np, model_names = submission.get_predictions_targets_models(leaderboard, update_nan_with_default=False, print_details=False)
                    predictions_np = np.copy(raw_predictions_np)
                    predictions_np[np.isnan(predictions_np)] = default_result


                    # Get full metric results
                    for metric_name, metric in leaderboard_metrics.items():
                        metric_output = metric.compute(predictions_np, raw_targets_np)

                        metadata = metric_output['metadata']

                        if metadata is not None:
                            if isinstance(metadata, dict):
                                for key, value in metadata.items():
                                    metrics[key] = value
                            else:
                                raise RuntimeError('Unexpected type for metadata: {}'.format(metadata))
                    time_str = time_utils.convert_epoch_to_iso(submission.submission_epoch)
                    new_data = {}
                    new_data['model_name'] = all_model_ids
                    new_data['team_name'] = [actor.name] * len(all_model_ids)
                    new_data['submission_timestamp'] = [time_str] * len(all_model_ids)
                    new_data['data_split'] = [data_split] * len(all_model_ids)
                    new_data['prediction'] = [float(i) for i in raw_predictions_np]
                    new_data['ground_truth'] = [float(i) for i in raw_targets_np]
                    for key, value in metrics.items():
                        data = [float(i) for i in value]
                        if len(data) == len(all_model_ids):
                            new_data[key] = data

                    new_df = pd.DataFrame(new_data)
                    all_dfs.append(new_df)

    result_df = pd.concat(all_dfs)

    result_df.to_csv(output_filepath, index=False)

    print('Finished writing round results to {}'.format(output_filepath))

    return result_df

def build_round_csvs(trojai_config: TrojaiConfig, leaderboard_names: list, output_dirpath: str, overwrite_csv=True):

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # Process both archived and active rounds
    active_rounds = trojai_config.active_leaderboard_names
    archive_rounds = trojai_config.archive_leaderboard_names

    all_rounds = list()
    all_rounds.extend(active_rounds)
    all_rounds.extend(archive_rounds)

    for round_name in all_rounds:
        leaderboard = Leaderboard.load_json(trojai_config, round_name)

        # Build the round CSV
        round_df = build_round_dataset_csv(leaderboard, output_dirpath, overwrite_csv=overwrite_csv)

        # Build round results
        build_round_results(trojai_config, leaderboard, round_df, output_dirpath, overwrite_csv=overwrite_csv)



if __name__ == '__main__':

    parser = configargparse.ArgumentParser(
        config_file_parser_class=JSONConfigFileParser,
        description='Builds the CSV files related to a round. Two CSVs are output, METADATA CSV describing the training and a RESULTS CSV describing the results of the round'
    )
    parser.add_argument('--config-filepath', is_config_file=True, help='The filepath to the config file.')
    parser.add_argument('--trojai-config-filepath', type=str, help='The file path to the trojai config', required=True)
    parser.add_argument('--leaderboard-names', nargs='*', help='The names of leaderboards to use, by default will use those specified in trojai config', default=[])
    parser.add_argument('--save-config-filepath', type=str, help='The path to save the config file.')
    parser.add_argument('--output-dirpath', type=str, help='Output dirpath for CSVs.', required=True)

    args = parser.parse_args()

    if args.save_config_filepath is not None:
        parser.write_config_file(args, [args.save_config_filepath])

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    build_round_csvs(trojai_config, args.leaderboard_names, args.output_dirpath)