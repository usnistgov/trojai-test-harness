import os

from actor_executor import json_io


class TrojaiConfig(object):

    TROJAI_CONFIG_FILENAME = 'trojai_config.json'

    def __init__(self, trojai_dirpath: str, init=False):
        self.trojai_dirpath = os.path.abspath(trojai_dirpath)
        self.html_repo_dirpath = os.path.join(self.trojai_dirpath, 'html')
        self.submission_dirpath = os.path.join(self.trojai_dirpath, 'submissions')
        self.datasets_dirpath = os.path.join(self.trojai_dirpath, 'datasets')
        self.results_dirpath = os.path.join(self.trojai_dirpath, 'results')
        self.leaderboard_configs_dirpath = os.path.join(self.trojai_dirpath, 'leaderboard-configs')
        self.actors_filepath = os.path.join(self.trojai_dirpath, 'actors.json')
        self.accepting_submissions = False
        self.active_leaderboard_names = list()

        if init:
            self.initialize_directories()

    def initialize_directories(self):
        os.makedirs(self.trojai_dirpath, exist_ok=True)
        os.makedirs(self.html_repo_dirpath, exist_ok=True)
        os.makedirs(self.submission_dirpath, exist_ok=True)
        os.makedirs(self.datasets_dirpath, exist_ok=True)
        os.makedirs(self.results_dirpath, exist_ok=True)
        os.makedirs(self.leaderboard_configs_dirpath, exist_ok=True)


    def __str__(self):
        msg = 'TrojaiConfig: (\n'
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

    def save_json(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.trojai_dirpath, TrojaiConfig.TROJAI_CONFIG_FILENAME)
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath):
        return json_io.read(filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Creates trojai config')
    parser.add_argument('--trojai-dirpath', type=str,
                        help='The main trojai directory path',
                        required=True)
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()

    trojai_config = TrojaiConfig(args.trojai_dirpath, args.init)
    trojai_config.save_json()
    print('Created: {}'.format(trojai_config))
