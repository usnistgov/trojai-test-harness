import os

from trojai_leaderboard import json_io


class TrojaiConfig(object):

    TROJAI_CONFIG_FILENAME = 'trojai_config.json'
    DEFAULT_VM_CONFIGURATION = {'gpu-vm-01': '192.168.200.2', 'gpu-vm-41': '192.168.200.3', 'gpu-vm-81': '192.168.200.4', 'gpu-vm-c1': '192.168.200.5'}

    def __init__(self, trojai_dirpath: str, token_pickle_filepath: str, slurm_execute_script_filepath: str, init=False, control_slurm_queue_name='control'):
        self.trojai_dirpath = os.path.abspath(trojai_dirpath)
        self.trojai_config_filepath = os.path.join(self.trojai_dirpath, TrojaiConfig.TROJAI_CONFIG_FILENAME)
        self.token_pickle_filepath = token_pickle_filepath
        self.html_repo_dirpath = os.path.join(self.trojai_dirpath, 'html')
        self.submission_dirpath = os.path.join(self.trojai_dirpath, 'submissions')
        self.datasets_dirpath = os.path.join(self.trojai_dirpath, 'datasets')
        self.results_dirpath = os.path.join(self.trojai_dirpath, 'results')
        self.leaderboard_configs_dirpath = os.path.join(self.trojai_dirpath, 'leaderboard-configs')
        self.actors_filepath = os.path.join(self.trojai_dirpath, 'actors.json')
        self.log_filepath = os.path.join(self.trojai_dirpath, 'trojai.log')
        self.slurm_execute_script_filepath = slurm_execute_script_filepath
        self.control_slurm_queue_name = control_slurm_queue_name
        self.accepting_submissions = False
        self.active_leaderboard_names = list()
        self.vms = TrojaiConfig.DEFAULT_VM_CONFIGURATION

        # TODO: Pass this in as a parameter?
        self.log_file_byte_limit = int(1 * 1024 * 1024)

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
    parser.add_argument('--token-pickle-filepath', type=str, help='The token pickle filepath', required=True)
    parser.add_argument('--slurm-execute-script-filepath', type=str, help='The filepath to the slurm execute script (trojai-test-harness/slurm_scripts/run_python.sh', required=True)
    parser.add_argument('--control-slurm-queue-name', type=str, help='The name of the slurm queue used for control', default='control')
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()

    trojai_config = TrojaiConfig(args.trojai_dirpath, args.token_pickle_filepath, args.slurm_execute_script_filepath, args.init, args.control_slurm_queue_name)
    trojai_config.save_json()
    print('Created: {}'.format(trojai_config))
