import logging
import os.path
import subprocess
import time
import typing
import glob
from typing import List
from leaderboards.mail_io import TrojaiMail
from leaderboards import jsonschema_checker
from leaderboards.dataset import Dataset
from leaderboards.trojai_config import TrojaiConfig


def check_gpu(host):
    if host == Task.LOCAL_VM_IP:
        child = subprocess.Popen(['nvidia-smi'])
    else:
        child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'nvidia-smi'])
    return child.wait()


def check_file_in_container(container_filepath, filepath_in_container):
    child = subprocess.Popen(['singularity', 'exec', container_filepath, 'test', '-f', filepath_in_container])
    return child.wait()


def check_dir_in_container(container_filepath, dirpath_in_container):
    child = subprocess.Popen(['singularity', 'exec', container_filepath, 'test', '-d', dirpath_in_container])
    return child.wait()


def cleanup_scratch(host, remote_scratch):
    if remote_scratch == '':
        logging.error('Failed to cleanup scratch, errors with passing path: {}, it must not be an empty string'.format(remote_scratch))
        return -1

    if host == Task.LOCAL_VM_IP:
        all_files = glob.glob('{}/*'.format(remote_scratch))
        child = subprocess.Popen(['rm', '-rf'] + all_files)
    else:
        child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'rm', '-rf', '{}/*'.format(remote_scratch)])
    return child.wait()


def create_directory_on_vm(host, dirpath: str):
    if host == Task.LOCAL_VM_IP:
        params = ['mkdir', '-p', dirpath]
    else:
        params = ['ssh', '-q', 'trojai@' + host, 'mkdir', '-p', dirpath]
    child = subprocess.Popen(params)
    return child.wait()

def rsync_file_to_vm(host, source_filepath, remote_path, source_params = [], remote_params = []):
    params = []
    if host == Task.LOCAL_VM_IP:
        params.extend(['rsync'])
    else:
        params.extend(['rsync', '-e', 'ssh -q'])

    params.extend(source_params)
    if host == Task.LOCAL_VM_IP:
        params.extend([source_filepath, remote_path])
    else:
        params.extend([source_filepath, 'trojai@' + host + ':' + remote_path])
    params.extend(remote_params)

    logging.debug(' '.join(params))

    rc = subprocess.run(params)
    return rc.returncode


def rsync_dir_to_vm(host, source_dirpath, remote_dirpath, source_params = [], remote_params = []):
    params = []
    if host == Task.LOCAL_VM_IP:
        params.extend(['rsync', '-ar', '--prune-empty-dirs', '--delete'])
    else:
        params.extend(['rsync', '-ar', '-e', 'ssh -q', '--prune-empty-dirs', '--delete'])
    params.extend(source_params)

    if host == Task.LOCAL_VM_IP:
        import shlex
        params.extend([shlex.quote(source_dirpath), shlex.quote(remote_dirpath)])
    else:
        params.extend([source_dirpath, 'trojai@' + host + ':' + remote_dirpath])
    params.extend(remote_params)

    logging.debug(' '.join(params))

    rc = subprocess.run(params)
    return rc.returncode


def scp_dir_from_vm(host, remote_dirpath, local_dirpath):
    logging.debug('remote: {} to {}'.format(remote_dirpath, local_dirpath))
    if host == Task.LOCAL_VM_IP:
        cmd = ['cp', '-r'] + glob.glob('{}/*'.format(remote_dirpath)) + [local_dirpath]
        # child = subprocess.Popen(cmd)
    else:
        cmd = ['scp', '-r', '-q', 'trojai@{}:{}/*'.format(host, remote_dirpath), local_dirpath]
        # child = subprocess.Popen(cmd)
    logging.info(' '.join(cmd))
    rc = subprocess.run(cmd)
    return rc.returncode
    # return child.wait()


def check_subprocess_error(sc, errors, msg, send_mail=False, subject=''):
    if sc != 0:
        message = '{}, status code: {}'.format(msg, sc)
        logging.error(message)

        if send_mail:
            TrojaiMail().send(to='trojai@nist.gov', subject=subject, message=message)

        return errors

    return ''


class Task(object):
    LOCAL_VM_IP = 'local'

    # TODO: Determine optimal init
    def __init__(self):
        pass

    def check_instance_params(self, trojai_config: TrojaiConfig):
        raise NotImplementedError()

    def get_remote_dataset_dirpath(self, remote_dirpath, leaderboard_name):
        raise NotImplementedError()

    def verify_dataset(self, leaderboard_name, dataset: Dataset, required_files: List[str]):
        raise NotImplementedError()

    def run_basic_checks(self, vm_ip, vm_name):
        raise NotImplementedError()

    def run_submission_checks(self, submission_filepath):
        raise NotImplementedError()

    def run_submission_schema_header_checks(self, submission_filepath):
        raise NotImplementedError()

    def copy_in_env(self, vm_ip, vm_name, trojai_config: TrojaiConfig, custom_remote_home: str=None, custom_remote_scratch: str=None):
        raise NotImplementedError()

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], custom_remote_home: str=None, custom_remote_scratch: str=None, custom_metaparameter_filepath: str=None):
        raise NotImplementedError()

    def execute_submission(self, vm_ip, vm_name, python_execution_env_filepath: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], info_dict: dict, custom_remote_home: str=None, custom_remote_scratch: str=None, custom_metaparameter_filepath: str=None, subset_model_ids: list=None, custom_result_dirpath: str=None):
        raise NotImplementedError()

    def get_basic_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str],  custom_remote_home: str, custom_remote_scratch: str , custom_metaparameter_filepath: str, subset_model_ids: list, custom_result_dirpath: str):
        raise NotImplementedError()

    def get_custom_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, custom_remote_home: str, custom_remote_scratch: str, custom_result_dirpath: str):
        raise NotImplementedError()

    def copy_out_results(self, vm_ip, vm_name, result_dirpath, custom_remote_home: str=None, custom_remote_scratch: str=None):
        raise NotImplementedError()

    def package_results(self, result_dirpath: str, info_dict: dict):
        raise NotImplementedError()

    def cleanup_vm(self, vm_ip, vm_name, custom_remote_home: str=None, custom_remote_scratch: str=None):
        raise NotImplementedError()

    def load_ground_truth(self, dataset: Dataset) -> typing.OrderedDict[str, float]:
        raise NotImplementedError()


class TrojAITask(Task):
    VALID_TECHNIQUE_TYPES = ['Weight Analysis', 'Trigger Inversion', 'Attribution Analysis', 'Jacobian Inspection', 'Other']

    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, task_type: str, evaluate_model_python_filepath: str = None,  remote_home: str = '/home/trojai', remote_scratch: str = '/mnt/scratch'):
        super().__init__()

        self.task_type = task_type

        self.default_prediction_result = 0.5

        self.remote_home = remote_home
        self.remote_dataset_dirpath: str = self.get_remote_dataset_dirpath(self.remote_home, leaderboard_name)
        self.remote_scratch = remote_scratch
        self.evaluate_model_python_filepath = evaluate_model_python_filepath

        task_dirpath = os.path.dirname(os.path.realpath(__file__))
        vm_scripts_dirpath = os.path.normpath(os.path.join(task_dirpath, '..', 'vm_scripts'))

        if self.evaluate_model_python_filepath is None:
            self.evaluate_model_python_filepath = os.path.join(vm_scripts_dirpath, 'evaluate_task.py')

    def check_instance_params(self, trojai_config: TrojaiConfig):
        has_updated = False
        if not hasattr(self, 'evaluate_model_python_filepath'):
            task_dirpath = os.path.dirname(os.path.realpath(__file__))
            vm_scripts_dirpath = os.path.normpath(os.path.join(task_dirpath, '..', 'vm_scripts'))
            self.evaluate_model_python_filepath = os.path.join(vm_scripts_dirpath, 'evaluate_task.py')
            has_updated = True

        if not hasattr(self, 'task_type'):
            self.task_type = 'TODO: Fix'
            has_updated = True

        return has_updated

    def get_remote_dataset_dirpath(self, remote_dirpath, leaderboard_name):
        return os.path.join(remote_dirpath, 'datasets', leaderboard_name)

    def verify_dataset(self, leaderboard_name, dataset: Dataset, required_files: List[str]):
        dataset_dirpath = dataset.dataset_dirpath
        source_dataset_dirpath = dataset.source_dataset_dirpath
        models_dirpath = os.path.join(dataset_dirpath, Dataset.MODEL_DIRNAME)

        is_valid = True

        if not os.path.exists(dataset_dirpath):
            logging.error('Failed to verify dataset {} for leaderboard: {}; dataset_dirpath {} does not exist '.format(dataset.dataset_name, leaderboard_name, dataset_dirpath))
            is_valid = False

        if source_dataset_dirpath is not None:
            if not os.path.exists(source_dataset_dirpath):
                logging.error('Failed to verify dataset {} for leaderboard: {}; source_dataset_dirpath {} does not exist, if it should not exist, then set the dirpath to None in the leaderboards config'.format(dataset.dataset_name, leaderboard_name, source_dataset_dirpath))
                is_valid = False

        if not os.path.exists(models_dirpath):
            logging.error('Failed to verify dataset {} for leaderboards: {}; models_dirpath {} does not exist '.format(dataset.dataset_name, leaderboard_name, models_dirpath))
            is_valid = False

        if is_valid:
            for model_id_dir in os.listdir(models_dirpath):
                for required_filename in required_files:
                    filepath = os.path.join(models_dirpath, str(model_id_dir), required_filename)
                    if not os.path.exists(filepath):
                        logging.error('Failed to verify dataset {} for leaderboards: {}; file in model {} does not exist '.format(dataset.dataset_name, leaderboard_name, filepath))
                        is_valid = False

        if is_valid:
            logging.info('dataset {} for leaderboards {} pass verification tests.'.format(dataset.dataset_name, leaderboard_name))
        return is_valid

    def run_basic_checks(self, vm_ip, vm_name):
        errors = ''
        logging.info('Checking GPU status')
        errors += check_subprocess_error(check_gpu(vm_ip), ':GPU:', '"{}" GPU may be off-line'.format(vm_name), send_mail=True, subject='VM "{}" GPU May be Offline'.format(vm_name))
        return errors

    def run_submission_checks(self, submission_filepath):
        errors = ''
        logging.info('Checking for parameters in container')

        # TODO: This might require modification for mitigation
        metaparameters_filepath = '/metaparameters.json'
        metaparameters_schema_filepath = '/metaparameters_schema.json'
        learned_parameters_dirpath = '/learned_parameters'

        sc = check_file_in_container(submission_filepath, metaparameters_filepath)
        errors += check_subprocess_error(sc, ':Container Parameters (metaparameters):', 'Metaparameters file "{}" not found in container'.format(metaparameters_filepath))

        sc = check_file_in_container(submission_filepath, metaparameters_schema_filepath)
        errors += check_subprocess_error(sc, ':Container Parameters (metaparameters schema):', 'Metaparameters schema file "{}" not found in container.'.format(metaparameters_schema_filepath))

        sc = check_dir_in_container(submission_filepath, learned_parameters_dirpath)
        errors += check_subprocess_error(sc, ':Container Parameters (learned parameters):', 'Learned parameters directory "{}" not found in container.'.format(learned_parameters_dirpath))

        logging.info('Running checks on jsonschema')

        # TODO: We will need to see how mitigation plays out for this
        if not jsonschema_checker.is_container_configuration_valid(submission_filepath):
            logging.error('Jsonschema contained errors.')
            errors += ':Container Parameters (jsonschema checker):'

        return errors

    def run_submission_schema_header_checks(self, submission_filepath):
        errors = ''

        # TODO: This might not be necessary for mitigation
        schema_dict = jsonschema_checker.collect_json_metaparams_schema(submission_filepath)

        default_title = 'Trojan Detection Container (trojai-example) - The template detector to be used within the TrojAI competition.'
        default_technique_description = 'Extracts model weights and trains a random forest regressor.'
        default_technique_changes = 'Output metaparameter.json file after reconfiguration'
        default_commit_id = ''
        default_repo_name = 'https://github.com/usnistgov/trojai-example'

        if 'title' in schema_dict:
            if default_title == schema_dict['title']:
                errors = ':Schema Header:'
                logging.warning('schema "title" is not valid')

        if 'technique_description' in schema_dict:
            if default_technique_description == schema_dict['technique_description']:
                errors = ':Schema Header:'
                logging.warning('schema "technique_description" is not valid')

        if 'technique_changes' in schema_dict:
            if default_technique_changes == schema_dict['technique_changes']:
                errors = ':Schema Header:'
                logging.warning('schema "technique_changes" is not valid')

        if 'commit_id' in schema_dict:
            if default_commit_id == schema_dict['commit_id']:
                errors = ':Schema Header:'
                logging.warning('schema "commit_id" is not valid')

        if 'repo_name' in schema_dict:
            if default_repo_name == schema_dict['repo_name']:
                errors = ':Schema Header:'
                logging.warning('schema "repo_name" is not valid')

        if 'technique_type' in schema_dict:
            if not isinstance(schema_dict['technique_type'], list):
                errors = ':Schema Header:'
                logging.warning('schema "technique_type" should be a list')
            if len(schema_dict['technique_type']) == 0:
                errors = ':Schema Header:'
                logging.warning('schema "technique_type" should have at least one item in it')

            for technique_type in schema_dict['technique_type']:
                if technique_type.upper() not in map(str.upper, TrojAITask.VALID_TECHNIQUE_TYPES):
                    errors = ':Schema Header:'
                    logging.warning('schema "technique_type" value "{}" is not found from list of valid technique types {}. Contact the TrojAI team to add your technique type.'.format(technique_type, Task.VALID_TECHNIQUE_TYPES))

        return errors

    def copy_in_env(self, vm_ip, vm_name, trojai_config: TrojaiConfig, custom_remote_home: str=None, custom_remote_scratch: str=None):
        logging.info("Copying miniconda3 env into VM.")
        remote_home = self.remote_home
        remote_scratch = self.remote_scratch

        if custom_remote_home is not None:
            remote_home = custom_remote_home

        if custom_remote_scratch is not None:
            remote_scratch = custom_remote_scratch

        errors = ''

        sc = rsync_dir_to_vm(vm_ip, trojai_config.local_trojai_conda_env, remote_home)
        errors += check_subprocess_error(sc, ':Copy in:', '{} failed to copy in conda env {}'.format(vm_name, trojai_config.local_trojai_conda_env))

        return errors

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], custom_remote_home: str=None, custom_remote_scratch: str=None, custom_metaparameter_filepath: str=None):
        logging.info('Copying in task data')

        remote_home = self.remote_home
        remote_scratch = self.remote_scratch

        if custom_remote_home is not None:
            remote_home = custom_remote_home

        if custom_remote_scratch is not None:
            remote_scratch = custom_remote_scratch
        errors = ''

        sc = create_directory_on_vm(vm_ip, remote_home)
        errors += check_subprocess_error(sc, ':Create directory:', '{} failed to create directory {}'.format(vm_name, remote_home), send_mail=True, subject='{} failed to create directory {}'.format(vm_name, remote_home))

        sc = create_directory_on_vm(vm_ip, remote_scratch)
        errors += check_subprocess_error(sc, ':Create directory:', '{} failed to create directory {}'.format(vm_name, remote_scratch), send_mail=True, subject='{} failed to create directory {}'.format(vm_name, remote_scratch))

        # copy in evaluate scripts (all models and single model) and update permissions
        permissions_params = ['--perms', '--chmod=u+rwx']
        sc = rsync_file_to_vm(vm_ip, self.evaluate_model_python_filepath, remote_home, source_params=permissions_params)
        errors += check_subprocess_error(sc, ':Copy in:', '{} evaluate python model script copy in may have failed'.format(vm_name), send_mail=True, subject='{} evaluate model script copy failed'.format(vm_name))

        # copy in submission filepath
        sc = rsync_file_to_vm(vm_ip, submission_filepath, remote_scratch)
        errors += check_subprocess_error(sc, ':Copy in:', '{} submission copy in may have failed'.format(vm_name), send_mail=True, subject='{} submission copy failed'.format(vm_name))

        if custom_metaparameter_filepath is not None:
            sc = rsync_file_to_vm(vm_ip, custom_metaparameter_filepath, remote_scratch)
            errors += check_subprocess_error(sc, ':Copy in (custom_metaparam):', '{} submission copy in may have failed'.format(vm_name),  send_mail=False, subject='{} submission copy failed'.format(vm_name))

        # Copy in datasets
        if vm_ip != Task.LOCAL_VM_IP:
            dataset_dirpath = dataset.dataset_dirpath
            source_dataset_dirpath = dataset.source_dataset_dirpath
            remote_dataset_dirpath = self.remote_dataset_dirpath

            # Create datasets dirpath
            sc = create_directory_on_vm(vm_ip, remote_dataset_dirpath)
            errors += check_subprocess_error(sc, ':Create directory:', '{} failed to create directory {}'.format(vm_name, remote_dataset_dirpath), send_mail=True, subject='{} failed to create directory {}'.format(vm_name, remote_dataset_dirpath))

            copy_dataset_params = ['--copy-links']

            # copy in round training dataset and source data
            if source_dataset_dirpath is not None:
                sc = rsync_dir_to_vm(vm_ip, source_dataset_dirpath, remote_dataset_dirpath, source_params=copy_dataset_params)
                errors += check_subprocess_error(sc, ':Copy in:', '{} source dataset copy in may have failed'.format(vm_name), send_mail=True, subject='{} source dataset copy failed'.format(vm_name))

            if training_dataset is not None:
                sc = rsync_dir_to_vm(vm_ip, training_dataset.dataset_dirpath, remote_dataset_dirpath, source_params=copy_dataset_params)
                errors += check_subprocess_error(sc, ':Copy in:', '{} training dataset copy in may have failed'.format(vm_name), send_mail=True, subject='{} training dataset copy failed'.format(vm_name))

            # copy in models
            source_params = []
            for excluded_file in excluded_files:
                source_params.append('--exclude={}'.format(excluded_file))
            source_params.extend(copy_dataset_params)
            sc = rsync_dir_to_vm(vm_ip, dataset_dirpath, remote_dataset_dirpath, source_params=source_params)
            errors += check_subprocess_error(sc, ':Copy in:', '{} model dataset {} copy in may have failed'.format(vm_name, dataset.dataset_name), send_mail=True, subject='{} dataset copy failed'.format(vm_name))

        return errors

    # TODO: It may be necessary to specify the python_execution_env_filepath related to the task, rather than the current use from the trojai config
    def execute_submission(self, vm_ip, vm_name, python_execution_env_filepath: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], info_dict: dict, custom_remote_home: str=None, custom_remote_scratch: str=None, custom_metaparameter_filepath: str=None, subset_model_ids: list=None, custom_result_dirpath: str=None):
        remote_home = self.remote_home
        remote_scratch = self.remote_scratch

        if custom_remote_home is not None:
            remote_home = custom_remote_home

        if custom_remote_scratch is not None:
            remote_scratch = custom_remote_scratch

        errors = ''
        remote_evaluate_models_python_filepath = os.path.join(remote_home, os.path.basename(self.evaluate_model_python_filepath))
        submission_name = os.path.basename(submission_filepath)

        start_time = time.time()
        logging.info('Starting execution of {}'.format(submission_name))

        if vm_ip == Task.LOCAL_VM_IP:
            params = ['timeout', '-s', 'SIGTERM', '-k', '30', str(dataset.timeout_time_sec) + 's', python_execution_env_filepath, remote_evaluate_models_python_filepath]
        else:
            params = ['ssh', '-q', 'trojai@' + vm_ip, 'timeout', '-s', 'SIGTERM', '-k', '30', str(dataset.timeout_time_sec) + 's', python_execution_env_filepath, remote_evaluate_models_python_filepath]

        params.extend(self.get_basic_execute_args(vm_ip, submission_filepath, dataset, training_dataset, excluded_files, custom_remote_home, custom_remote_scratch, custom_metaparameter_filepath, subset_model_ids, custom_result_dirpath))
        params.extend(self.get_custom_execute_args(vm_ip, submission_filepath, dataset, training_dataset, custom_remote_home, custom_remote_scratch, custom_result_dirpath))

        logging.info('Launching with params {}'.format(' '.join(params)))

        rc = subprocess.run(params)
        execute_status = rc.returncode

        execution_time = time.time() - start_time
        logging.info('Submission: {}, runtime: {} seconds'.format(submission_name, execution_time))
        logging.info('Execute statues: {}'.format(execute_status))

        if execute_status == -9 or execute_status == 124 or execute_status == (128+9):
            logging.error('VM {} execute submission {} timed out.'.format(vm_name, submission_name))
            errors += ':Timeout:'
        elif execute_status != 0:
            logging.error('VM {} execute submission {} may have failed.'.format(vm_name, submission_name))
            errors += ':Execute:'

        info_dict['execution_runtime'] = execution_time

        return errors

    def get_basic_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str],  custom_remote_home: str, custom_remote_scratch: str , custom_metaparameter_filepath: str, subset_model_ids: list, custom_result_dirpath: str):
        remote_home = self.remote_home
        remote_scratch = self.remote_scratch
        remote_training_dataset_dirpath = None

        if custom_remote_home is not None:
            remote_home = custom_remote_home

        if custom_remote_scratch is not None:
            remote_scratch = custom_remote_scratch

        if vm_ip == Task.LOCAL_VM_IP:
            remote_models_dirpath = os.path.join(dataset.dataset_dirpath, Dataset.MODEL_DIRNAME)
            if training_dataset is not None:
                remote_training_dataset_dirpath = training_dataset.dataset_dirpath
        else:
            remote_models_dirpath = os.path.join(self.remote_dataset_dirpath, dataset.dataset_name, Dataset.MODEL_DIRNAME)

            if training_dataset is not None:
                remote_training_dataset_dirpath = os.path.join(self.remote_dataset_dirpath, training_dataset.dataset_name)

        submission_name = os.path.basename(submission_filepath)
        remote_submission_filepath = os.path.join(remote_scratch, submission_name)
        remote_submission_filepath = "'{}'".format(remote_submission_filepath)
        result_dirpath = os.path.join(remote_scratch, 'results')

        if custom_result_dirpath is not None:
            result_dirpath = custom_result_dirpath

        args = ['--models-dirpath', remote_models_dirpath, '--task-type', self.task_type, '--submission-filepath', remote_submission_filepath, '--home-dirpath', remote_home, '--scratch-dirpath', remote_scratch,
                '--result-dirpath', result_dirpath]

        if remote_training_dataset_dirpath is not None:
            args.extend(['--training-dataset-dirpath', remote_training_dataset_dirpath])

        # Add excluded files into list
        args.append('--rsync-excludes')
        for excluded_file in excluded_files:
            args.append(excluded_file)

        if custom_metaparameter_filepath is not None:
            metaparameter_name = os.path.basename(custom_metaparameter_filepath)
            metaparameter_filepath = os.path.join(remote_scratch, metaparameter_name)
            args.append('--metaparameter-filepath')
            args.append(metaparameter_filepath)

        if subset_model_ids is not None:
            args.append('--subset-model-id')
            for subset_model_id in subset_model_ids:
                args.append(subset_model_id)

        if dataset.source_dataset_dirpath is not None:
            if vm_ip == Task.LOCAL_VM_IP:
                remote_source_data_dirpath = dataset.source_dataset_dirpath
            else:
                source_data_dirname = os.path.basename(dataset.source_dataset_dirpath)
                remote_source_data_dirpath = os.path.join(self.remote_dataset_dirpath, source_data_dirname)
            args.extend(['--source-dataset-dirpath', remote_source_data_dirpath])

        return args

    def get_custom_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, custom_remote_home: str, custom_remote_scratch: str, custom_result_dirpath: str):
        return []

    def copy_out_results(self, vm_ip, vm_name, result_dirpath, custom_remote_home: str=None, custom_remote_scratch: str=None):
        remote_home = self.remote_home
        remote_scratch = self.remote_scratch

        if custom_remote_home is not None:
            remote_home = custom_remote_home

        if custom_remote_scratch is not None:
            remote_scratch = custom_remote_scratch

        logging.info('Copying out results')
        errors = ''
        remote_result_dirpath = os.path.join(remote_scratch, 'results')
        sc = scp_dir_from_vm(vm_ip, remote_result_dirpath, result_dirpath)
        errors += check_subprocess_error(sc, ':Copy Out:', 'Copy out results may have failed for VM {}'.format(vm_name))
        return errors

    # TODO: This might have to be customized
    def package_results(self, result_dirpath: str, info_dict: dict):
        model_prediction_dict = {}
        for model_prediction_file_name in os.listdir(result_dirpath):
            if not model_prediction_file_name.startswith('id-') or model_prediction_file_name.endswith('-walltime.txt'):
                continue

            model_name = model_prediction_file_name.replace('.txt', '')
            model_prediction_filepath = os.path.join(result_dirpath, model_prediction_file_name)

            if not os.path.exists(model_prediction_filepath):
                continue

            try:
                with open(model_prediction_filepath, 'r') as prediction_fh:
                    file_contents = prediction_fh.readline().strip()
                    model_prediction_dict[model_name] = float(file_contents)
            except:
                pass

        info_dict['predictions'] = model_prediction_dict

    def cleanup_vm(self, vm_ip, vm_name, custom_remote_home: str=None, custom_remote_scratch: str=None):
        remote_home = self.remote_home
        remote_scratch = self.remote_scratch

        if custom_remote_home is not None:
            remote_home = custom_remote_home

        if custom_remote_scratch is not None:
            remote_scratch = custom_remote_scratch

        errors = ''
        logging.info('Performing VM cleanup.')
        sc = cleanup_scratch(vm_ip, remote_scratch)
        errors += check_subprocess_error(sc, ':Cleanup:', '{} cleanup failed with status code {}'.format(vm_name, sc))
        return errors

class MitigationTask(TrojAITask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, task_type: str,
                 task_script_filepath: str = None, remote_home: str = '/home/trojai',
                 remote_scratch: str = '/mnt/scratch'):
        super().__init__(trojai_config, leaderboard_name, task_type, task_script_filepath, remote_home, remote_scratch)

    def run_submission_checks(self, submission_filepath):
        return ''

    def run_submission_schema_header_checks(self, submission_filepath):
        return ''

    # copy_in_env, use base
    # cleanup_vm, use base

    def copy_in_env(self, vm_ip, vm_name, trojai_config: TrojaiConfig, custom_remote_home: str = None,
                    custom_remote_scratch: str = None):
        logging.info("Copying miniconda3 env into VM.")
        remote_home = self.remote_home
        remote_scratch = self.remote_scratch

        if custom_remote_home is not None:
            remote_home = custom_remote_home

        if custom_remote_scratch is not None:
            remote_scratch = custom_remote_scratch

        errors = ''

        sc = rsync_dir_to_vm(vm_ip, trojai_config.local_trojai_conda_env, remote_home)
        errors += check_subprocess_error(sc, ':Copy in:', '{} failed to copy in conda env {}'.format(vm_name,
                                                                                                     trojai_config.local_trojai_conda_env))

        return errors

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset,
                          training_dataset: Dataset, excluded_files: List[str], custom_remote_home: str = None,
                          custom_remote_scratch: str = None, custom_metaparameter_filepath: str = None):
        logging.info('Copying in task data')

        remote_home = self.remote_home
        remote_scratch = self.remote_scratch

        if custom_remote_home is not None:
            remote_home = custom_remote_home

        if custom_remote_scratch is not None:
            remote_scratch = custom_remote_scratch
        errors = ''

        sc = create_directory_on_vm(vm_ip, remote_home)
        errors += check_subprocess_error(sc, ':Create directory:',
                                         '{} failed to create directory {}'.format(vm_name, remote_home),
                                         send_mail=True,
                                         subject='{} failed to create directory {}'.format(vm_name, remote_home))

        sc = create_directory_on_vm(vm_ip, remote_scratch)
        errors += check_subprocess_error(sc, ':Create directory:',
                                         '{} failed to create directory {}'.format(vm_name, remote_scratch),
                                         send_mail=True,
                                         subject='{} failed to create directory {}'.format(vm_name, remote_scratch))

        # copy in evaluate scripts (all models and single model) and update permissions
        permissions_params = ['--perms', '--chmod=u+rwx']
        sc = rsync_file_to_vm(vm_ip, self.evaluate_model_python_filepath, remote_home,
                              source_params=permissions_params)
        errors += check_subprocess_error(sc, ':Copy in:',
                                         '{} evaluate python model script copy in may have failed'.format(vm_name),
                                         send_mail=True,
                                         subject='{} evaluate model script copy failed'.format(vm_name))

        # copy in submission filepath
        sc = rsync_file_to_vm(vm_ip, submission_filepath, remote_scratch)
        errors += check_subprocess_error(sc, ':Copy in:', '{} submission copy in may have failed'.format(vm_name),
                                         send_mail=True, subject='{} submission copy failed'.format(vm_name))

        # Copy in datasets
        if vm_ip != Task.LOCAL_VM_IP:
            dataset_dirpath = dataset.dataset_dirpath
            source_dataset_dirpath = dataset.source_dataset_dirpath
            remote_dataset_dirpath = self.remote_dataset_dirpath

            # Create datasets dirpath
            sc = create_directory_on_vm(vm_ip, remote_dataset_dirpath)
            errors += check_subprocess_error(sc, ':Create directory:',
                                             '{} failed to create directory {}'.format(vm_name,
                                                                                       remote_dataset_dirpath),
                                             send_mail=True,
                                             subject='{} failed to create directory {}'.format(vm_name,
                                                                                               remote_dataset_dirpath))

            copy_dataset_params = ['--copy-links']

            if training_dataset is not None:
                sc = rsync_dir_to_vm(vm_ip, training_dataset.dataset_dirpath, remote_dataset_dirpath,
                                     source_params=copy_dataset_params)
                errors += check_subprocess_error(sc, ':Copy in:',
                                                 '{} training dataset copy in may have failed'.format(vm_name),
                                                 send_mail=True,
                                                 subject='{} training dataset copy failed'.format(vm_name))

            # copy in models
            source_params = []
            for excluded_file in excluded_files:
                source_params.append('--exclude={}'.format(excluded_file))
            source_params.extend(copy_dataset_params)
            sc = rsync_dir_to_vm(vm_ip, dataset_dirpath, remote_dataset_dirpath, source_params=source_params)
            errors += check_subprocess_error(sc, ':Copy in:',
                                             '{} model dataset {} copy in may have failed'.format(vm_name,
                                                                                                  dataset.dataset_name),
                                             send_mail=True, subject='{} dataset copy failed'.format(vm_name))

            return errors
        # def run_submission_checks(self, submission_filepath):
    #     errors = ''
    #     return errors
    #
    # def run_submission_schema_header_checks(self, submission_filepath):
    #     errors = ''
    #     return errors

def package_results(self, result_dirpath: str, info_dict: dict):
    pass
class ImageClassificationMitigationTask(MitigationTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, task_script_filepath=None):
        super().__init__(trojai_config, leaderboard_name, 'image_classification_mitigation', task_script_filepath)


class ImageTask(TrojAITask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, task_script_filepath=None):
        super().__init__(trojai_config, leaderboard_name, 'image', task_script_filepath)


class CyberTask(TrojAITask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, task_script_filepath=None):
       super().__init__(trojai_config, leaderboard_name, 'cyber', task_script_filepath)

class ReinforcementLearningTask(TrojAITask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, task_script_filepath=None):
        super().__init__(trojai_config, leaderboard_name, 'rl', task_script_filepath)

class NaturalLanguageProcessingTask(TrojAITask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str, task_script_filepath=None):
        self.tokenizers_dirpath = os.path.join(trojai_config.datasets_dirpath, leaderboard_name, 'tokenizers')
        super().__init__(trojai_config, leaderboard_name, 'nlp', task_script_filepath)

    def get_custom_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, custom_remote_home: str, custom_remote_scratch: str, custom_result_dirpath: str):
        if vm_ip == Task.LOCAL_VM_IP:
            remote_tokenizer_dirpath = self.tokenizers_dirpath
        else:
            tokenizer_dirname = os.path.basename(self.tokenizers_dirpath)
            remote_tokenizer_dirpath = os.path.join(self.remote_dataset_dirpath, tokenizer_dirname)
        return ['--tokenizer-dir', remote_tokenizer_dirpath]

    def verify_dataset(self, leaderboard_name, dataset: Dataset, required_files: List[str]):
        if not os.path.exists(self.tokenizers_dirpath):
            logging.error('Failed to verify dataset {} for leaderboards: {}; tokenizers_dirpath {} does not exist '.format(dataset.dataset_name, leaderboard_name, self.tokenizers_dirpath))
            return False

        return super().verify_dataset(leaderboard_name, dataset, required_files)

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], custom_remote_home: str=None, custom_remote_scratch: str=None, custom_metaparameter_filepath: str=None):
        errors = super().copy_in_task_data(vm_ip, vm_name, submission_filepath, dataset, training_dataset, excluded_files, custom_remote_home, custom_remote_scratch, custom_metaparameter_filepath)

        # Copy in tokenizers
        copy_dataset_params = ['--copy-links']
        sc = rsync_dir_to_vm(vm_ip, self.tokenizers_dirpath, self.remote_dataset_dirpath, source_params=copy_dataset_params)
        errors += check_subprocess_error(sc, ':Copy in:', '{} tokenizers copy in may have failed'.format(vm_name), send_mail=True, subject='{} tokenizers copy failed'.format(vm_name))

        return errors


class ImageSummary(ImageTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)


class ImageClassification(ImageTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)


class ImageObjectDetection(ImageTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)


class ImageSegmentation(ImageTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)


class NaturalLanguageProcessingSummary(NaturalLanguageProcessingTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)


class CausalLanguageModeling(TrojAITask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name, 'clm')


class NaturalLanguageProcessingSentiment(NaturalLanguageProcessingTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)


class NaturalLanguageProcessingNamedEntityRecognition(NaturalLanguageProcessingTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)


class NaturalLanguageProcessingQuestionAnswering(NaturalLanguageProcessingTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)

class CyberApkMalware(CyberTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)

class CyberPdfMalware(TrojAITask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        self.scale_params_filepath = os.path.join(trojai_config.datasets_dirpath, leaderboard_name, 'scale_params.npy')
        super().__init__(trojai_config, leaderboard_name, 'cyber_pdf')


    def get_custom_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, custom_remote_home: str, custom_remote_scratch: str, custom_result_dirpath: str):
        if vm_ip == Task.LOCAL_VM_IP:
            remote_scale_params_filepath = self.scale_params_filepath
        else:
            scale_params_dirname = os.path.basename(self.scale_params_filepath)
            remote_scale_params_filepath = os.path.join(self.remote_dataset_dirpath, scale_params_dirname)
        return ['--scale-params-filepath', remote_scale_params_filepath]

    def verify_dataset(self, leaderboard_name, dataset: Dataset, required_files: List[str]):
        if not os.path.exists(self.scale_params_filepath):
            logging.error('Failed to verify dataset {} for leaderboards: {}; scale_params_filepath {} does not exist '.format(dataset.dataset_name, leaderboard_name, self.scale_params_filepath))
            return False

        return super().verify_dataset(leaderboard_name, dataset, required_files)

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], custom_remote_home: str=None, custom_remote_scratch: str=None, custom_metaparameter_filepath: str=None):
        errors = super().copy_in_task_data(vm_ip, vm_name, submission_filepath, dataset, training_dataset, excluded_files, custom_remote_home, custom_remote_scratch, custom_metaparameter_filepath)

        # Copy in scale params
        sc = rsync_file_to_vm(vm_ip, self.scale_params_filepath, self.remote_dataset_dirpath)
        errors += check_subprocess_error(sc, ':Copy in:', '{} scale_params_filepath copy in may have failed'.format(vm_name), send_mail=True, subject='{} scale_params_filepath copy failed'.format(vm_name))

        return errors


class ReinforcementLearningLavaWorld(ReinforcementLearningTask):
    def __init__(self, trojai_config: TrojaiConfig, leaderboard_name: str):
        super().__init__(trojai_config, leaderboard_name)