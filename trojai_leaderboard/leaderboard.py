import os
import numpy as np

from trojai_leaderboard.trojai_config import TrojaiConfig
from trojai_leaderboard import json_io
from trojai_leaderboard.dataset import DatasetManager
from trojai_leaderboard.metrics import *
from trojai_leaderboard.tasks import *


class Leaderboard(object):
    INFO_FILENAME = 'info.json'
    TRAIN_DATASET_NAME = 'train'
    DEFAULT_DATASET_SPLIT_NAMES = ['train', 'test', 'sts', 'holdout']
    DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES = ['train', 'test', 'sts']
    VALID_TASK_NAMES = {'image_summary': ImageSummary,
                        'nlp_summary': NaturalLanguageProcessingSummary,
                        'image_classification': ImageClassification,
                        'image_object_detection' : ImageObjectDetection,
                        'image_segmentation': ImageSegmentation,
                        'nlp_sentiment': NaturalLanguageProcessingSentiment,
                        'nlp_ner': NaturalLanguageProcessingNamedEntityRecognition,
                        'nlp_qa': NaturalLanguageProcessingQuestionAnswering}


    GENERAL_SLURM_QUEUE_NAME = 'es'
    STS_SLURM_QUEUE_NAME = 'sts'

    def __init__(self, name: str, task_name: str, trojai_config: TrojaiConfig, add_default_data_split: bool = False):
        self.name = name
        self.task_name = task_name

        if self.task_name not in Leaderboard.VALID_TASK_NAMES:
            raise RuntimeError('Invalid task name: {}'.format(self.task_name))

        self.task = Leaderboard.VALID_TASK_NAMES[self.task_name](trojai_config, self.name)

        self.submission_dirpath = os.path.join(trojai_config.submission_dirpath, self.name)
        self.submissions_filepath = os.path.join(self.submission_dirpath, 'submissions.json')

        self.dataset_manager = DatasetManager()

        if add_default_data_split:
            for split_name in Leaderboard.DEFAULT_DATASET_SPLIT_NAMES:
                if split_name in Leaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES:
                    can_submit = True
                else:
                    can_submit = False

                if split_name == 'sts':
                    slurm_queue_name = Leaderboard.STS_SLURM_QUEUE_NAME
                    slurm_priority = 0
                else:
                    slurm_queue_name = Leaderboard.GENERAL_SLURM_QUEUE_NAME
                    slurm_priority = 0

                self.add_dataset(trojai_config, split_name, can_submit, slurm_queue_name, slurm_priority)

        self.initialize_directories()

    def get_submission_metrics(self, data_split_name):
        return self.dataset_manager.get_submission_metrics(data_split_name)

    def get_default_prediction_result(self):
        return self.task.default_prediction_result

    def get_dataset(self, data_split_name):
        return self.dataset_manager.get(data_split_name)

    def get_ground_truth_dirpath(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.groundtruth_dirpath

    def get_result_dirpath(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.results_dirpath

    def get_slurm_queue_name(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.slurm_queue_name

    def get_timeout_window_time(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.timeout_time_sec

    def can_submit_to_dataset(self, data_split_name: str):
        return self.dataset_manager.can_submit_to_dataset(data_split_name)

    def get_submission_data_split_names(self):
        return self.dataset_manager.get_submission_dataset_split_names()

    def add_dataset(self, trojai_config: TrojaiConfig, split_name: str, can_submit: bool, slurm_queue_name: str, slurm_priority: int, has_source_data: bool):
        if self.dataset_manager.has_dataset(split_name):
            raise RuntimeError('Dataset already exists in DatasetManager: {}'.format(split_name))

        dataset = Dataset(trojai_config, self.name, split_name, can_submit, slurm_queue_name, slurm_priority, has_source_data)
        if self.task.verify_dataset(self.name, dataset):
            self.dataset_manager.add_dataset(dataset)
            return True
        return False

    def initialize_directories(self):
        os.makedirs(self.submission_dirpath, exist_ok=True)
        self.dataset_manager.initialize_directories()

    def get_task(self) -> Task:
        return self.task

    def __str__(self):
        msg = 'Leaderboard: (\n'
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

    def save_json(self, trojai_config: TrojaiConfig):
        filepath = os.path.join(trojai_config.leaderboard_configs_dirpath, '{}_config.json'.format(self.name))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        json_io.write(filepath, self)

    @staticmethod
    def load_json(trojai_config: TrojaiConfig, name) -> 'Leaderboard':
        leaderboard_config = json_io.read(os.path.join(trojai_config.leaderboard_configs_dirpath, '{}_config.json'.format(name)))
        assert leaderboard_config.task_name in Leaderboard.VALID_TASK_NAMES
        return leaderboard_config

def init_leaderboard(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    leaderboard = Leaderboard(args.name, args.task_name, trojai_config, add_default_data_split=args.add_default_datasplit)
    leaderboard.save_json(trojai_config)
    print('Created: {}'.format(leaderboard))

def add_dataset_to_leaderboard(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    if args.slurm_queue_name is None:
        slurm_queue_name = Leaderboard.GENERAL_SLURM_QUEUE_NAME
    else:
        slurm_queue_name = args.slurm_queue_name

    leaderboard = Leaderboard.load_json(trojai_config, args.name)
    if leaderboard.add_dataset(trojai_config, args.split_name, args.can_submit, slurm_queue_name, args.slurm_priority, args.has_source_data):
        leaderboard.save_json(trojai_config)

        print('Added dataset {} to {}'.format(args.split_name, args.name))

    print('Failed to add dataset')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Runs leaderboard commands')
    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    init_parser = subparser.add_parser('init')
    init_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    init_parser.add_argument('--name', type=str, help='The name of the leaderboard', required=True)
    init_parser.add_argument('--task-name', type=str, choices=Leaderboard.VALID_TASK_NAMES, help='The name of the task for this leaderboard', required=True)
    init_parser.add_argument('--add-default-datasplit', help='Will attempt to add the default data splits: {}, if they fail task checks then will not be added. Need to call add-dataset when they are ready.'.format(Leaderboard.DEFAULT_DATASET_SPLIT_NAMES), action='store_true')
    init_parser.set_defaults(func=init_leaderboard)

    add_dataset_parser = subparser.add_parser('add-dataset')
    add_dataset_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    add_dataset_parser.add_argument('--name', type=str, help='The name of the leaderboard', required=True)
    add_dataset_parser.add_argument('--split-name', type=str, help='The dataset split name', required=True)
    add_dataset_parser.add_argument('--has-source-data', action='store_true', help='Indicates that the dataset has source data that is saved on disk, format: "leaderboard_name-source_data"')
    add_dataset_parser.add_argument('--can-submit', action='store_true', help='Whether actors can submit to the dataset')
    add_dataset_parser.add_argument('--slurm-queue-name', type=str, help='The name of the slurm queue')
    add_dataset_parser.add_argument('--slurm-priority', type=int, help='The priority when launching jobs for this dataset', default=0)
    add_dataset_parser.set_defaults(func=add_dataset_to_leaderboard)

    args = parser.parse_args()

    args.func(args)

