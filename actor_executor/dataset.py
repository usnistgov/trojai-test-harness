from actor_executor.trojai_config import TrojaiConfig

import os

class Dataset(object):
    DATASET_SUFFIX = 'dataset'

    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, split_name: str, can_submit: bool):
        self.dataset_name = self.get_dataset_name(leaderboard_name, split_name)
        self.split_name = split_name
        self.dataset_dirpath = os.path.join(trojai_config.datasets_dirpath, self.dataset_name)
        self.results_dirpath = os.path.join(trojai_config.results_dirpath, self.dataset_name)
        self.can_submit = can_submit

    def get_dataset_name(self, leaderboard_name, split_name):
        return  '{}-{}-{}'.format(leaderboard_name, split_name, Dataset.DATASET_SUFFIX)

    def initialize_directories(self):
        os.makedirs(self.dataset_dirpath, exist_ok=True)
        os.makedirs(self.results_dirpath, exist_ok=True)

    def verify(self):
        # TODO: Add other things to verify, may also create sub-classes based on task type if we want to verify existance of special things (like tokenizers)
        dataset_model_dirpath = os.path.join(self.dataset_dirpath, 'models')
        dataset_metadata_filepath = os.path.join(self.dataset_dirpath, 'METADATA.xml')

        if not os.path.exists(dataset_model_dirpath):
            raise RuntimeError('Unable to find model dirpath for dataset: {}'.format(dataset_model_dirpath))

        if not os.path.exists(dataset_metadata_filepath):
            raise RuntimeError('Unable to find metadata filepath for dataset: {}'.format(dataset_metadata_filepath))


    def __str__(self):
        msg = "Dataset: \n"
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg


class DatasetManager(object):

    def __init__(self):
        self.datasets = {}

    def __str__(self):
        msg = "Datasets: \n"
        for dataset in self.datasets.values():
            msg = msg + "  " + dataset.__str__() + "\n"
        return msg

    def add_dataset(self, trojai_config: TrojaiConfig, leaderboard_name: str, split_name: str, can_submit: bool):
        if split_name in self.datasets.keys():
            raise RuntimeError('Dataset already exists in DatasetManager: {}'.format(split_name))

        dataset = Dataset(trojai_config, leaderboard_name, split_name, can_submit)
        self.datasets[split_name] = dataset
        dataset.initialize_directories()
        print('Created: {}'.format(dataset))

    def get(self, split_name) -> Dataset:
        if split_name in self.datasets.keys():
            return self.datasets[split_name]
        else:
            raise RuntimeError('Invalid key in DatasetManager: {}'.format(split_name))

    def initialize_directories(self):
        for dataset in self.datasets.values():
            dataset.initialize_directories()
