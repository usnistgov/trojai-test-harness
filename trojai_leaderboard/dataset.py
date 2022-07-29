from trojai_leaderboard.trojai_config import TrojaiConfig
from trojai_leaderboard.metrics import *
import os

class Dataset(object):
    ALL_METRICS = [AverageCrossEntropy, CrossEntropyConfidenceInterval, BrierScore, ConfusionMatrix, ROC_AUC]
    BUFFER_TIME = 900
    # 300 models, 3 minutes per model + 15 minute buffer
    DEFAULT_TIMEOUT_SEC = 180 * 300 + BUFFER_TIME
    DEFAULT_STS_TIMEOUT_SEC = 180 * 10 + BUFFER_TIME
    DATASET_SUFFIX = 'dataset'
    DATASET_GROUNDTRUTH_NAME = 'groundtruth'
    MODEL_DIRNAME = 'models'

    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, split_name: str, can_submit: bool, slurm_queue_name: str, slurm_priority: int, timeout_time_per_model_sec: int=180, excluded_files=None):
        self.dataset_name = self.get_dataset_name(leaderboard_name, split_name)

        self.split_name = split_name
        self.dataset_dirpath = os.path.join(trojai_config.datasets_dirpath, self.dataset_name)
        self.results_dirpath = os.path.join(trojai_config.results_dirpath, self.dataset_name)
        self.groundtruth_dirpath = os.path.join(self.dataset_dirpath, Dataset.DATASET_GROUNDTRUTH_NAME)
        self.can_submit = can_submit
        self.slurm_queue_name = slurm_queue_name
        self.slurm_priority = slurm_priority
        self.excluded_files = excluded_files
        if self.excluded_files is None:
            self.excluded_files = ['detailed_stats.csv', 'detailed_config.json', 'ground_truth.csv', 'log.txt', 'machine.log', 'poisoned-example-data.json', 'stats.json']

        self.submission_metrics = dict()

        for metric in Dataset.ALL_METRICS:
            metric_inst = metric()
            self.submission_metrics[metric_inst.get_name()] = metric_inst

        model_dirpath = os.path.join(self.dataset_dirpath, 'models')
        if os.path.exists(model_dirpath):
            num_models = len([name for name in os.listdir(model_dirpath) if os.path.isdir(os.path.join(model_dirpath, name))])
            self.timeout_time_sec = num_models * timeout_time_per_model_sec
        else:
            if self.split_name == 'sts':
                self.timeout_time_sec = Dataset.DEFAULT_TIMEOUT_SEC
            else:
                self.timeout_time_sec = Dataset.DEFAULT_STS_TIMEOUT_SEC

    def get_dataset_name(self, leaderboard_name, split_name):
        return '{}-{}-{}'.format(leaderboard_name, split_name, Dataset.DATASET_SUFFIX)

    def initialize_directories(self):
        os.makedirs(self.dataset_dirpath, exist_ok=True)
        os.makedirs(self.results_dirpath, exist_ok=True)

    def get_ground_truth_dirpath(self):
        return self.groundtruth_dirpath

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

    def can_submit_to_dataset(self, data_split_name: str):
        if data_split_name in self.datasets.keys():
            dataset = self.datasets[data_split_name]
            return dataset.can_submit
        return False

    def get_submission_metrics(self, data_split_name: str) -> dict:
        dataset = self.get(data_split_name)
        return dataset.submission_metrics

    def get_submission_dataset_split_names(self):
        result = []
        for data_split_name, dataset in self.datasets.items():
            if dataset.can_submit:
                result.append(data_split_name)

        return result

    def add_dataset(self, trojai_config: TrojaiConfig, leaderboard_name: str, split_name: str, can_submit: bool, slurm_queue_name: str, slurm_priority: int, timeout_time_sec: int):
        if split_name in self.datasets.keys():
            raise RuntimeError('Dataset already exists in DatasetManager: {}'.format(split_name))

        dataset = Dataset(trojai_config, leaderboard_name, split_name, can_submit, slurm_queue_name, slurm_priority, timeout_time_sec)
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