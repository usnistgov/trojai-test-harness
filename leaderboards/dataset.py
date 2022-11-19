from leaderboards.trojai_config import TrojaiConfig
from leaderboards.metrics import *
import os

class Dataset(object):
    ALL_METRICS = [AverageCrossEntropy, CrossEntropyConfidenceInterval, BrierScore, ROC_AUC]
    BUFFER_TIME = 900
    # 300 models, 3 minutes per model + 15 minute buffer
    DEFAULT_TIMEOUT_SEC = 180 * 300 + BUFFER_TIME
    DEFAULT_STS_TIMEOUT_SEC = 180 * 10 + BUFFER_TIME
    DATASET_SUFFIX = 'dataset'
    # DATASET_GROUNDTRUTH_NAME = 'groundtruth'
    MODEL_DIRNAME = 'models'
    SOURCE_DATA_NAME = 'source-data'
    METADATA_NAME = 'METADATA.csv'
    GROUND_TRUTH_NAME = 'ground_truth.csv'

    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, split_name: str, can_submit: bool, slurm_queue_name: str, slurm_nice: int, has_source_data: bool, timeout_time_per_model_sec: int=600, excluded_files=None):
        self.split_name = split_name
        self.dataset_name = self.get_dataset_name()
        self.dataset_dirpath = os.path.join(trojai_config.datasets_dirpath, leaderboard_name, self.dataset_name)
        self.results_dirpath = os.path.join(trojai_config.results_dirpath, '{}-dataset'.format(leaderboard_name), self.dataset_name)
        self.can_submit = can_submit
        self.slurm_queue_name = slurm_queue_name
        self.slurm_nice = slurm_nice
        self.excluded_files = excluded_files
        self.source_dataset_dirpath = None
        if has_source_data:
            self.source_dataset_dirpath = os.path.join(trojai_config.datasets_dirpath, leaderboard_name, '{}'.format(Dataset.SOURCE_DATA_NAME))

        if self.excluded_files is None:
            self.excluded_files = copy.deepcopy(trojai_config.default_excluded_files)

        self.required_files = copy.deepcopy(trojai_config.default_required_files)

        if self.split_name == 'sts':
            self.auto_delete_submission = True
        else:
            self.auto_delete_submission = False

        self.auto_execute_split_names = []

        if self.split_name == 'test':
            self.auto_execute_split_names.append('train')

        self.submission_metrics = dict()

        for metric in Dataset.ALL_METRICS:
            metric_inst = metric()
            self.submission_metrics[metric_inst.get_name()] = metric_inst

        self.evaluation_metric_name = 'Cross Entropy'

        num_models = self.get_num_models()

        self.submission_window_time_sec = Dataset.BUFFER_TIME
        if num_models > 0:
            self.timeout_time_sec = num_models * timeout_time_per_model_sec
        else:
            if self.split_name == 'sts':
                self.timeout_time_sec = Dataset.DEFAULT_STS_TIMEOUT_SEC
            else:
                self.timeout_time_sec = Dataset.DEFAULT_TIMEOUT_SEC

    def add_metric(self, metric: Metric):
        self.submission_metrics[metric.get_name()] = metric

    def remove_metric(self, metric_name):
        if metric_name not in self.submission_metrics:
            print('Failed to remove {}, it does not exist for {}'.format(metric_name, self.split_name))
        else:
            del self.submission_metrics[metric_name]

    def refresh_metrics(self):
        new_metrics = {}
        for metric in self.submission_metrics.values():
            new_metrics[metric.get_name()] = metric
        self.submission_metrics = new_metrics

    def get_num_models(self):
        model_dirpath = os.path.join(self.dataset_dirpath, Dataset.MODEL_DIRNAME)
        if os.path.exists(model_dirpath):
            return len([name for name in os.listdir(model_dirpath) if os.path.isdir(os.path.join(model_dirpath, name))])
        else:
            return 0

    def get_dataset_name(self):
        return '{}-{}'.format(self.split_name, Dataset.DATASET_SUFFIX)

    def initialize_directories(self):
        os.makedirs(self.dataset_dirpath, exist_ok=True)
        os.makedirs(self.results_dirpath, exist_ok=True)

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

    def add_metric(self, data_split_name, metric: Metric):
        if data_split_name is None:
            for dataset in self.datasets.values():
                metric_copy = copy.deepcopy(metric)
                dataset.add_metric(metric_copy)
        else:
            dataset = self.get(data_split_name)
            dataset.add_metric(metric)

    def refresh_metrics(self):
        for dataset in self.datasets.values():
            dataset.refresh_metrics()

    def remove_metric(self, data_split_name, metric_name):
        if data_split_name is None:
            for dataset in self.datasets.values():
                dataset.remove_metric(metric_name)
        else:
            dataset = self.get(data_split_name)
            dataset.remove_metric(metric_name)

    def has_dataset(self, split_name: str):
        return split_name in self.datasets.keys()

    def add_dataset(self, dataset: Dataset):
        if dataset.split_name in self.datasets.keys():
            raise RuntimeError('Dataset already exists in DatasetManager: {}'.format(dataset.dataset_name))

        self.datasets[dataset.split_name] = dataset
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
