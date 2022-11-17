import os
import jsonpickle
from leaderboards import json_io


def get_round10_reduced_config(config_dict):
    reduced_config = dict()

    config_data_location = config_dict
    if 'py/state' in config_dict:
        config_data_location = config_dict['py/state']

    reduced_config['model_architecture'] = config_data_location['model_architecture']
    reduced_config['source_dataset'] = config_data_location['source_dataset']

    return reduced_config

def get_round1_2_3_reduced_config(config_dict):
    reduced_config = dict()

    config_data_location = config_dict
    if 'py/state' in config_dict:
        config_data_location = config_dict['py/state']

    reduced_config['model_architecture'] = config_data_location['MODEL_ARCHITECTURE']
    return reduced_config

def get_round4_reduced_config(config_dict):
    reduced_config = dict()

    config_data_location = config_dict
    if 'py/state' in config_dict:
        config_data_location = config_dict['py/state']

    reduced_config['model_architecture'] = config_data_location['model_architecture']
    return reduced_config

def create_reduced_config(args):
    datasets_dirpath = args.datasets_dirpath
    round_names = args.round_names
    data_split_names = args.data_split
    create_empty_config = args.create_empty_config

    if not os.path.exists(datasets_dirpath):
        print('Unable to locate dataset dirpath: {}'.format(datasets_dirpath))
        return
    for round_name in round_names:
        dataset_dirpath = os.path.join(datasets_dirpath, round_name)
        if not os.path.exists(dataset_dirpath):
            print('Unable to locate round dataset: {}'.format(dataset_dirpath))

        for data_split_name in data_split_names:
            dataset_models_dirpath = os.path.join(dataset_dirpath, '{}-{}'.format(round_name, data_split_name), 'models')

            print('Updating {}'.format(dataset_models_dirpath))
            if not os.path.exists(dataset_models_dirpath):
                print('Unable to locate data split dirpath: {}'.format(dataset_models_dirpath))
                continue


            for model_name in os.listdir(dataset_models_dirpath):
                dataset_model_dirpath = os.path.join(dataset_models_dirpath, model_name)
                reduced_config_filepath = os.path.join(dataset_model_dirpath, 'reduced-config.json')


                if os.path.exists(reduced_config_filepath):
                    print('Reduced config file already exists: {}'.format(reduced_config_filepath))

                if create_empty_config:
                    with open(reduced_config_filepath, 'w') as fp:
                        pass
                else:
                    config_filepath = os.path.join(dataset_model_dirpath, 'config.json')
                    config_dict = json_io.read(config_filepath)
                    reduced_config = None
                    if 'round10' in round_name:
                        reduced_config = get_round10_reduced_config(config_dict)
                    if 'round1' in round_name or 'round2' in round_name or 'round3' in round_name:
                        reduced_config = get_round1_2_3_reduced_config(config_dict)
                    if 'round4' in round_name:
                        reduced_config = get_round4_reduced_config(config_dict)
                    else:
                        raise RuntimeError('Must implement function to get {} reduced configuration', round_name)

                    if reduced_config is not None:
                        with open(reduced_config_filepath, 'w') as fp:
                            fp.write(jsonpickle.encode(reduced_config, warn=True, indent=2))
                    else:
                        raise RuntimeError('reduced config was not defined for {}'.format(reduced_config_filepath))











if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Converts old datasets to new example data format')

    parser.add_argument('--datasets-dirpath', type=str, help='The directory path for the datasets')
    parser.add_argument('--round-names', nargs='+', help='The names of the rounds to update')
    parser.add_argument('--data-split', nargs='+', help='The names of the datasplit to update')
    parser.add_argument('--create-empty-config', action='store_true', help='Whether to just create empty configs')

    args = parser.parse_args()

    create_reduced_config(args)
