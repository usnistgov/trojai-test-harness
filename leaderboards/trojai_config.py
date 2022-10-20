import os

from leaderboards import json_io


class TrojaiConfig(object):

    TROJAI_CONFIG_FILENAME = 'trojai_config.json'
    DEFAULT_VM_CONFIGURATION = {'gpu-vm-01': '192.168.200.2', 'gpu-vm-41': '192.168.200.3', 'gpu-vm-81': '192.168.200.4', 'gpu-vm-c1': '192.168.200.5'}

    def __init__(self, trojai_dirpath: str, token_pickle_filepath: str, slurm_execute_script_filepath: str=None, init=False, control_slurm_queue_name='control'):
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
        file_dirpath = os.path.dirname(os.path.realpath(__file__))
        self.trojai_test_harness_dirpath = os.path.normpath(os.path.join(file_dirpath, '..'))
        self.task_evaluator_script_filepath = os.path.join(file_dirpath, 'task_executor.py')
        self.python_env = '/home/trojai/trojai-env/bin/python3'
        self.default_excluded_files = ["detailed_stats.csv", "config.json", "ground_truth.csv", "log.txt", "log-per-class.txt", "machine.log", "poisoned-example-data", "stats.json", "METADATA.csv", "trigger_*", "DATA_LICENSE.txt", "METADATA_DICTIONARY.csv", "models-packaged", "README.txt"]
        self.default_required_files = ["model.pt", "ground_truth.csv", "clean-example-data", "reduced-config.json"]
        self.leaderboard_csvs_dirpath = os.path.join(self.datasets_dirpath, 'leaderboard_summary_csvs')
        self.vm_cpu_cores_per_partition = {'es': 10, 'sts': 10}
        self.job_color_key = {604800: 'success-color-dark text-light',
                              1209600: 'warning-color-dark',
                              float('inf'): 'danger-color-dark text-light'}

        self.send_global_metrics = False


        self.slurm_execute_script_filepath = slurm_execute_script_filepath
        if slurm_execute_script_filepath is None:
            file_dirpath = os.path.dirname(os.path.realpath(__file__))
            slurm_scripts_dirpath = os.path.join(file_dirpath, '..', 'slurm_scripts')
            self.slurm_execute_script_filepath = os.path.normpath(os.path.join(slurm_scripts_dirpath, 'run_python.sh'))

        self.control_slurm_queue_name = control_slurm_queue_name
        self.accepting_submissions = False
        self.active_leaderboard_names = list()
        self.archive_leaderboard_names = list()
        self.html_default_leaderboard_name = ''
        self.vms = TrojaiConfig.DEFAULT_VM_CONFIGURATION


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
        os.makedirs(self.leaderboard_csvs_dirpath, exist_ok=True)


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
    def load_json(filepath) -> 'TrojaiConfig':
        return json_io.read(filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Creates trojai config')
    parser.add_argument('--trojai-dirpath', type=str,
                        help='The main trojai directory path',
                        required=True)
    parser.add_argument('--token-pickle-filepath', type=str, help='The token pickle filepath', required=True)
    parser.add_argument('--control-slurm-queue-name', type=str, help='The name of the slurm queue used for control', default='control')
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()

    trojai_config = TrojaiConfig(trojai_dirpath=args.trojai_dirpath, token_pickle_filepath=args.token_pickle_filepath, init=args.init, control_slurm_queue_name=args.control_slurm_queue_name)
    trojai_config.save_json()
    print('Created: {}'.format(trojai_config))
