import os
import shutil

def convert_old_dataset_example_data(args):
    datasets_dirpath = args.datasets_dirpath
    round_names = args.round_names
    data_split_names = args.data_split

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


            for model_name in os.listdir(dataset_model_dirpath):
                dataset_model_dirpath = os.path.join(dataset_models_dirpath, model_name)

                if not os.path.exists(dataset_model_dirpath):
                    print('Unable to locate model: {}'.format(dataset_model_dirpath))
                    continue


                clean_types = ['example_data', 'clean_example_data', 'clean-example-data.json']
                poisoned_types = ['poisoned_example_data', 'poisoned-example-data.json']

                new_clean_dirpath = os.path.join(dataset_model_dirpath, 'clean-example-data')
                new_poisoned_dirpath = os.path.join(dataset_model_dirpath, 'poisoned-example-data')

                for clean_type in clean_types:
                    clean_path = os.path.join(dataset_model_dirpath, clean_type)

                    if os.path.exists(clean_path):
                        if os.path.isdir(clean_path):
                            # move clean path to clean-example-data
                            shutil.move(clean_path, new_clean_dirpath)
                        elif os.path.isfile(clean_path):
                            # create clean-example-data and move clean_path into it
                            os.makedirs(new_clean_dirpath, exist_ok=True)
                            shutil.move(clean_path, new_clean_dirpath)
                        else:
                            print('Unknown path: {}'.format(clean_path))

                for poisoned_type in poisoned_types:
                    poison_path = os.path.join(dataset_model_dirpath, poisoned_type)

                    if os.path.exists(poison_path):
                        if os.path.isdir(poison_path):
                            # move poison_path to poisoned-example-data
                            shutil.move(poison_path, new_poisoned_dirpath)
                        elif os.path.isfile(poison_path):
                            # create poisoned-example-data and move poison_path into it
                            os.makedirs(new_poisoned_dirpath, exist_ok=True)
                            shutil.move(poison_path, new_poisoned_dirpath)
                            pass
                        else:
                            print('Unknown path: {}'.format(poison_path))








if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Converts old datasets to new example data format')

    parser.add_argument('--datasets-dirpath', type=str, help='The directory path for the datasets')
    parser.add_argument('--round-names', nargs='+', help='The names of the rounds to update')
    parser.add_argument('--data-split', nargs='+', help='The names of the datasplit to update')

    args = parser.parse_args()

    convert_old_dataset_example_data(args)
