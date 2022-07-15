import os
from actor_executor.trojai_config import TrojaiConfig
from actor_executor import json_io
from actor_executor.dataset import DatasetManager


class Leaderboard(object):

    DEFAULT_DATASET_SPLIT_NAMES = ['train', 'test', 'sts', 'holdout']
    DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES = ['train', 'test', 'sts']
    VALID_TASK_NAMES = ['image_summary', 'nlp_summary', 'image_classification', 'image_object_detection',
                        'image_segmentation', 'nlp_sentiment', 'nlp_ner', 'nlp_qa']

    # 15 minute timeout
    STS_TIMEOUT_TIME_SEC = 900

    def __init__(self, name: str, task_name: str, trojai_config: TrojaiConfig, init=False, timeout_time_sec: int=259200):
        self.name = name
        self.timeout_time_sec = timeout_time_sec
        self.task_name = task_name

        if self.task_name not in Leaderboard.VALID_TASK_NAMES:
            raise RuntimeError('Invalid task name: {}'.format(self.task_name))

        self.submission_dirpath = os.path.join(trojai_config.submission_dirpath, self.name)
        self.submissions_filepath = os.path.join(self.submission_dirpath, 'submissions.json')

        self.dataset_manager = DatasetManager()

        for split_name in Leaderboard.DEFAULT_DATASET_SPLIT_NAMES:
            if split_name in Leaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES:
                can_submit = True
            else:
                can_submit = False
            self.dataset_manager.add_dataset(trojai_config, self.name, split_name, can_submit)

        if init:
            self.initialize_directories()

    def add_dataset(self, trojai_config: TrojaiConfig, split_name: str, can_submit: bool):
        self.dataset_manager.add_dataset(trojai_config, self.name, split_name, can_submit)

    def get_timeout_window_time(self, data_split_name):
        if data_split_name == 'sts':
            return Leaderboard.STS_TIMEOUT_TIME_SEC
        else:
            return self.timeout_time_sec

    def initialize_directories(self):
        os.makedirs(self.submission_dirpath, exist_ok=True)
        self.dataset_manager.initialize_directories()

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
    def load_json(trojai_config: TrojaiConfig, dataset_name) -> 'Leaderboard':
        leaderboard_config = json_io.read(os.path.join(trojai_config.leaderboard_configs_dirpath, '{}_config.json'.format(dataset_name)))
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

    parser.add_argument('--add-dataset', help='Adds dataset to leaderboard in CSV format: "leaderboard_name,split_name,can_submit"')

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    if args.add_dataset is not None:
        items = args.add_dataset.split(',')
        if len(items) != 3:
            raise RuntimeError('Invalid number of arguments for adding dataset')
        leaderboard_name = items[0]
        split_name = items[1]
        can_submit = items[2].lower() == 'true'

        leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
        leaderboard.add_dataset(trojai_config, split_name, can_submit)
        leaderboard.save_json(trojai_config)

        print('Added dataset {} to {}'.format(split_name, leaderboard_name))

    else:
        leaderboard = Leaderboard(args.name, args.task_name, trojai_config, init=True, timeout_time_sec=args.timeout)

        leaderboard.save_json(trojai_config)

        print('Created: {}'.format(leaderboard))
