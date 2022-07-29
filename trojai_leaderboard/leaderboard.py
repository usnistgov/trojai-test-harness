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
    VALID_TASK_NAMES = {'image_summary' : ImageSummary(),
                        'nlp_summary': NaturalLanguageProcessingSummary(),
                        'image_classification': ImageClassification(),
                        'image_object_detection' : ImageObjectDetection(),
                        'image_segmentation': ImageSegmentation(),
                        'nlp_sentiment': NaturalLanguageProcessingSentiment(),
                        'nlp_ner': NaturalLanguageProcessingNamedEntityRecognition(),
                        'nlp_qa': NaturalLanguageProcessingQuestionAnswering()}


    # 15 minute timeout
    STS_TIMEOUT_TIME_SEC = 900

    GENERAL_SLURM_QUEUE_NAME = 'general'
    STS_SLURM_QUEUE_NAME = 'sts'

    def __init__(self, name: str, task_name: str, trojai_config: TrojaiConfig, init=False, timeout_time_sec: int=259200):
        self.name = name
        self.timeout_time_sec = timeout_time_sec
        self.task_name = task_name

        if self.task_name not in Leaderboard.VALID_TASK_NAMES:
            raise RuntimeError('Invalid task name: {}'.format(self.task_name))

        self.task = Leaderboard.VALID_TASK_NAMES[self.task_name]

        self.submission_dirpath = os.path.join(trojai_config.submission_dirpath, self.name)
        self.submissions_filepath = os.path.join(self.submission_dirpath, 'submissions.json')

        self.dataset_manager = DatasetManager()

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

            self.dataset_manager.add_dataset(trojai_config, self.name, split_name, can_submit, slurm_queue_name, slurm_priority)

        if init:
            self.initialize_directories()

    def get_submission_metrics(self, data_split_name):
        return self.dataset_manager.get_submission_metrics(data_split_name)

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

    def can_submit_to_dataset(self, data_split_name: str):
        return self.dataset_manager.can_submit_to_dataset(data_split_name)

    def get_submission_data_split_names(self):
        return self.dataset_manager.get_submission_dataset_split_names()

    def add_dataset(self, trojai_config: TrojaiConfig, split_name: str, can_submit: bool, slurm_queue_name: str, slurm_priority: int):
        self.dataset_manager.add_dataset(trojai_config, self.name, split_name, can_submit, slurm_queue_name, slurm_priority)

    def get_timeout_window_time(self, data_split_name):
        if data_split_name == 'sts':
            return Leaderboard.STS_TIMEOUT_TIME_SEC
        else:
            return self.timeout_time_sec

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Creates a leaderboard config')
    parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    parser.add_argument('--name', type=str, help='The name of the leaderboard', required=True)
    parser.add_argument('--timeout', type=int, help='The timeout time in seconds for the leaderboard', default=259200)
    parser.add_argument('--task-name', type=str, choices=Leaderboard.VALID_TASK_NAMES,
                        help='The name of the task for this leaderboard', required=True)
    parser.add_argument('--init', action='store_true')

    parser.add_argument('--add-dataset', help='Adds dataset to leaderboard in CSV format: "leaderboard_name,split_name,can_submit,slurm_priority"')

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    if args.add_dataset is not None:
        items = args.add_dataset.split(',')
        if len(items) != 5:
            raise RuntimeError('Invalid number of arguments for adding dataset')
        leaderboard_name = items[0]
        split_name = items[1]
        can_submit = items[2].lower() == 'true'
        slurm_queue_name = items[3]
        slurm_priority = int(items[4])

        leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
        leaderboard.add_dataset(trojai_config, split_name, can_submit, slurm_queue_name, slurm_priority)
        leaderboard.save_json(trojai_config)

        print('Added dataset {} to {}'.format(split_name, leaderboard_name))

    else:
        leaderboard = Leaderboard(args.name, args.task_name, trojai_config, init=True, timeout_time_sec=args.timeout)

        leaderboard.save_json(trojai_config)

        print('Created: {}'.format(leaderboard))
