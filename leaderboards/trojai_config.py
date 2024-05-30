import json
import os

from leaderboards import json_io
from python_utils import update_object_values
from typing import Dict

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
        self.leaderboard_results_dirpath = os.path.join(self.trojai_dirpath, 'leaderboard-results')
        self.actors_filepath = os.path.join(self.trojai_dirpath, 'actors.json')
        self.log_filepath = os.path.join(self.trojai_dirpath, 'trojai.log')
        file_dirpath = os.path.dirname(os.path.realpath(__file__))
        self.trojai_test_harness_dirpath = os.path.normpath(os.path.join(file_dirpath, '..'))
        self.task_evaluator_script_filepath = os.path.join(file_dirpath, 'task_executor.py')
        self.python_env = '/home/trojai/trojai-env/bin/python3'
        self.evaluate_python_env = '/home/trojai/miniconda3/envs/trojai_evaluate/bin/python'
        self.local_trojai_conda_env = '/home/trojai/miniconda3'
        # self.default_excluded_files = ["detailed_stats.csv", "detailed_timing_stats.csv", "config.json", "ground_truth.csv", "log.txt", "log-per-class.txt", "machine.log", "poisoned-example-data", "stats.json", "METADATA.csv", "trigger_*", "DATA_LICENSE.txt", "METADATA_DICTIONARY.csv", "models-packaged", "README.txt", "watermark.json"]
        # self.default_required_files = ["model.pt", "ground_truth.csv", "clean-example-data", "reduced-config.json"]
        self.leaderboard_csvs_dirpath = os.path.join(self.datasets_dirpath, 'leaderboard_summary_csvs')
        self.vm_cpu_cores_per_partition = {'es': 10, 'sts': 10}
        self.job_color_key = {604800: 'text-success font-weight-bold',
                              1209600: 'text-warning font-weight-bold',
                              float('inf'): 'text-danger font-weight-bold'}

        self.summary_metric_email_addresses = []
        self.summary_metrics_dirpath = os.path.join(trojai_dirpath, 'summary_metrics')
        self.summary_metric_update_timeframe = 3600
        self.last_summary_metric_update = 0

        self.slurm_execute_script_filepath = slurm_execute_script_filepath
        if slurm_execute_script_filepath is None:
            file_dirpath = os.path.dirname(os.path.realpath(__file__))
            slurm_scripts_dirpath = os.path.join(file_dirpath, '..', 'slurm_scripts')
            self.slurm_execute_script_filepath = os.path.normpath(os.path.join(slurm_scripts_dirpath, 'run_python.sh'))

        self.control_slurm_queue_name = control_slurm_queue_name
        self.commit_and_push_html = True
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
        os.makedirs(self.summary_metrics_dirpath, exist_ok=True)

    def can_apply_summary_updates(self, cur_epoch):
        if self.last_summary_metric_update + self.summary_metric_update_timeframe <= cur_epoch:
            self.last_summary_metric_update = cur_epoch
            self.save_json()
            return True
        return False

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

# Updates the configuration file to map to the latest version. Only used when swithing to new infrastructure
def update_configuration_latest(args):
    trojai_config_filepath = args.trojai_config_filepath

    backup_filepath = os.path.join(os.path.dirname(trojai_config_filepath), 'trojai_config_backup.json')
    old_trojai_config = None

    with open(trojai_config_filepath, 'r') as fp:
        old_trojai_config = json.load(fp)

    # Save backup
    if os.path.exists(backup_filepath):
        print('Error, backup already exists, cancelling')
        return

    with open(backup_filepath, 'w') as fp:
        json.dump(old_trojai_config, fp, indent=2)

    # Init from dictionary
    trojai_dirpath = None
    token_pickle_filepath = None
    slurm_execute_script_filepath = None
    control_slurm_queue_name = None
    if 'trojai_dirpath' in old_trojai_config:
        trojai_dirpath = old_trojai_config['trojai_dirpath']
    if 'token_pickle_filepath' in old_trojai_config:
        token_pickle_filepath = old_trojai_config['token_pickle_filepath']
    if 'slurm_execute_script_filepath' in old_trojai_config:
        slurm_execute_script_filepath = old_trojai_config['slurm_execute_script_filepath']
    if 'control_slurm_queue_name' in old_trojai_config:
        control_slurm_queue_name = old_trojai_config['control_slurm_queue_name']

    if trojai_dirpath is None or token_pickle_filepath is None or slurm_execute_script_filepath is None or control_slurm_queue_name is None:
        print(
            'Error: unable to initialize from dictionary, missing entries. Make sure you are passing the correct config')
        return

    new_config = TrojaiConfig(trojai_dirpath=trojai_dirpath, token_pickle_filepath=token_pickle_filepath,
                              slurm_execute_script_filepath=slurm_execute_script_filepath, init=False,
                              control_slurm_queue_name=control_slurm_queue_name)

    # Delete py/object prior to update
    if 'py/object' in old_trojai_config:
        del old_trojai_config['py/object']

    update_object_values(new_config, old_trojai_config)

    new_config.save_json()
    print('Finished converting trojai config')


def init_cmd(args):
    trojai_config = TrojaiConfig(trojai_dirpath=args.trojai_dirpath, token_pickle_filepath=args.token_pickle_filepath,
                                 init=args.init, control_slurm_queue_name=args.control_slurm_queue_name)
    trojai_config.save_json()
    print('Created: {}'.format(trojai_config))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Creates trojai config')
    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)


    init_parser = subparser.add_parser('init')
    init_parser.add_argument('--trojai-dirpath', type=str,
                        help='The main trojai directory path',
                        required=True)
    init_parser.add_argument('--token-pickle-filepath', type=str, help='The token pickle filepath', required=True)
    init_parser.add_argument('--control-slurm-queue-name', type=str, help='The name of the slurm queue used for control', default='control')
    init_parser.add_argument('--init', action='store_true')
    init_parser.set_defaults(func=init_cmd)

    update_config_parser = subparser.add_parser('update-config')
    update_config_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    update_config_parser.set_defaults(func=update_configuration_latest)


    args = parser.parse_args()

    args.func(args)
