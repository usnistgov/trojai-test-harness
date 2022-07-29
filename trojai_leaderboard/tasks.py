import logging
import os.path
import subprocess
import time
from trojai_leaderboard.mail_io import TrojaiMail
from trojai_leaderboard import jsonschema_checker
from trojai_leaderboard.dataset import Dataset


def check_gpu(host):
    child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'nvidia-smi'])
    return child.wait()


def check_file_in_container(container_filepath, filepath_in_container):
    child = subprocess.Popen(['singularity', 'exec', container_filepath, 'test', '-f', filepath_in_container])
    return child.wait()


def check_dir_in_container(container_filepath, dirpath_in_container):
    child = subprocess.Popen(['singularity', 'exec', container_filepath, 'test', '-d', dirpath_in_container])
    return child.wait()


def cleanup_scratch(host):
    child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'rm', '-rf', '/mnt/scratch/*'])
    return child.wait()


def rsync_file_to_vm(host, source_filepath, remove_path, source_params = [], remote_params = []):
    params = []
    params.extend(['rsync'])
    params.extend(source_params)
    params.extend([source_filepath, 'trojai@'+host+':\"' + remove_path + '\"'])
    params.extend(remote_params)

    child = subprocess.Popen(params)
    return child.wait()


def rsync_dir_to_vm(host, source_dirpath, remote_dirpath, source_params = [], remote_params = []):
    params = []
    params.extend(['rsync', '-ar', '-e', 'ssh -q', '--prune-empty-dirs', '--delete'])
    params.extend(source_params)
    params.extend([source_dirpath, 'trojai@' + host + ':\"' + remote_dirpath + '\"'])
    params.extend(remote_params)

    child = subprocess.Popen(params)
    return child.wait()


def scp_dir_from_vm(host, remote_dirpath, source_dirpath):
    child = subprocess.Popen(['scp', '-r', '-q', 'trojai@{}:{}/*'.format(host, remote_dirpath), source_dirpath])
    return child.wait()


def check_subprocess_error(sc, msg, errors, send_mail=False, subject=''):
    if sc != 0:
        message = '{}, status code: {}'.format(msg, sc)
        logging.error(message)

        if send_mail:
            TrojaiMail().send(to='trojai@nist.gov', subject=subject, message=message)

        return errors

    return ''


class Task(object):
    def __init__(self, task_script_filepath: str, source_data_name: str = 'source_data', evaluate_models_filepath: str = None,
                 evaluate_model_filepath: str = None, remote_home: str = '/home/trojai', remote_scratch: str = '/mnt/scratch'):
        self.evaluate_models_filepath = evaluate_models_filepath
        self.evaluate_model_filepath = evaluate_model_filepath
        self.task_script_filepath = task_script_filepath
        self.source_data_name = source_data_name

        self.remote_home = remote_home
        self.remote_scratch = remote_scratch

        task_dirpath = os.path.dirname(os.path.realpath(__file__))
        vm_scripts_dirpath = os.path.join(task_dirpath, '..', 'vm_scripts')

        if self.evaluate_models_filepath is None:
            self.evaluate_models_filepath = os.path.join(vm_scripts_dirpath, 'evaluate_models.sh')

        if self.evaluate_model_filepath is None:
            self.evaluate_model_filepath = os.path.join(vm_scripts_dirpath, 'evaluate_model.sh')

    def run_basic_checks(self, vm_ip, vm_name):
        errors = ''
        logging.info('Checking GPU status')
        errors += check_subprocess_error(check_gpu(vm_ip), ':GPU:', '"{}" GPU may be off-line'.format(vm_name), send_mail=True, subject='VM "{}" GPU May be Offline'.format(vm_name))
        return errors

    def run_submission_checks(self, submission_filepath):
        errors = ''
        logging.info('Checking for parameters in container')

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

        if not jsonschema_checker.is_container_configuration_valid(submission_filepath):
            logging.error('Jsonschema contained errors.')
            errors += ':Container Parameters (jsonschema checker):'

        return errors

    def cleanup_vm(self, vm_ip, vm_name):
        errors = ''
        logging.info('Performing VM cleanup.')
        sc = cleanup_scratch(vm_ip)
        errors += check_subprocess_error(sc, ':Cleanup:', '{} cleanup failed with status code {}'.format(vm_name, sc))
        return errors

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset):
        logging.info('Copying in task data')
        errors = ''
        # copy in evaluate scripts (all models and single model) and update permissions
        permissions_params = ['--perms', '--chmod=u+rwx']
        sc = rsync_file_to_vm(vm_ip, self.evaluate_model_filepath, self.remote_home, source_params=permissions_params)
        errors += check_subprocess_error(sc, ':Copy in:', '{} evaluate model script copy in may have failed'.format(vm_name), send_mail=True, subject='{} evaluate model script copy failed'.format(vm_name))
        sc = rsync_file_to_vm(vm_ip, self.task_script_filepath, self.remote_home, source_params=permissions_params)
        errors += check_subprocess_error(sc, ':Copy in:', '{} evaluate models script copy in may have failed'.format(vm_name), send_mail=True, subject='{} evaluate models script copy failed'.format(vm_name))

        # copy in submission filepath
        sc = rsync_file_to_vm(vm_ip, submission_filepath, self.remote_scratch)
        errors += check_subprocess_error(sc, ':Copy in:', '{} submission copy in may have failed'.format(vm_name), send_mail=True, subject='{} submission copy failed'.format(vm_name))

        dataset_dirpath = dataset.dataset_dirpath
        source_dataset_dirpath = os.path.join(dataset_dirpath, self.source_data_name)

        # copy in round training dataset and source data
        sc = rsync_dir_to_vm(vm_ip, source_dataset_dirpath, self.remote_home)
        errors += check_subprocess_error(sc, ':Copy in:', '{} source dataset copy in may have failed'.format(vm_name), send_mail=True, subject='{} source dataset copy failed'.format(vm_name))
        sc = rsync_dir_to_vm(vm_ip, training_dataset.dataset_dirpath, self.remote_home)
        errors += check_subprocess_error(sc, ':Copy in:', '{} training dataset copy in may have failed'.format(vm_name), send_mail=True, subject='{} training dataset copy failed'.format(vm_name))

        # copy in models
        models_dirpath = os.path.join(dataset_dirpath, Dataset.MODEL_DIRNAME)
        excluded_files_str = ','.join(dataset.excluded_files)
        source_params = ['--exclude={' + excluded_files_str + '}']
        sc = rsync_dir_to_vm(vm_ip, models_dirpath, self.remote_home, source_params=source_params)
        errors += check_subprocess_error(sc, ':Copy in:', '{} model dataset {} copy in may have failed'.format(vm_name, dataset.dataset_name), send_mail=True, subject='{} dataset copy failed'.format(vm_name))

        return errors

    def execute_submission(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, info_dict: dict):
        errors = ''
        remote_evaluate_models_filepath = os.path.join(self.remote_home, os.path.basename(self.evaluate_models_filepath))
        submission_name = os.path.basename(submission_filepath)
        task_script_name = os.path.basename(self.task_script_filepath)

        start_time = time.time()
        logging.info('Starting execution of {}.'.format(submission_name))
        child = subprocess.Popen(['ssh', '-q', 'trojai@'+vm_ip, 'timeout', '-s', 'SIGTERM', '-k', '30', dataset.timeout_time_sec + 's', remote_evaluate_models_filepath, '\"{}\"'.format(submission_name), task_script_name])
        execute_status = child.wait()
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

    def copy_out_results(self, vm_ip, vm_name, result_dirpath):
        logging.info('Copying out results')
        errors = ''
        remote_result_dirpath = os.path.join(self.remote_scratch, 'results')
        sc = scp_dir_from_vm(vm_ip, remote_result_dirpath, result_dirpath)
        errors += check_subprocess_error(sc, ':Copy Out:', 'Copy out results may have failed for VM {}'.format(vm_name))
        return errors

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


class ImageTask(Task):
    def __init__(self, task_script_filepath=None):
        if task_script_filepath is None:
            task_dirpath = os.path.dirname(os.path.realpath(__file__))
            task_scripts_dirpath = os.path.join(task_dirpath, '..', 'vm_scripts')
            task_script_filepath = os.path.join(task_scripts_dirpath, 'image_task.sh')
        super().__init__(task_script_filepath)


class NaturalLanguageProcessingTask(Task):
    def __init__(self, task_script_filepath=None):
        if task_script_filepath is None:
            task_dirpath = os.path.dirname(os.path.realpath(__file__))
            task_scripts_dirpath = os.path.join(task_dirpath, '..', 'task_scripts')
            task_script_filepath = os.path.join(task_scripts_dirpath, 'nlp_task.sh')
        super().__init__(task_script_filepath)

        self.tokenizer_dirname = 'tokenizers'

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset):
        errors = super().copy_in_task_data(vm_ip, vm_name, submission_filepath, dataset, training_dataset)

        # Copy in tokenizers
        tokenizers_dirpath = os.path.join(dataset.dataset_dirpath, self.tokenizer_dirname)
        sc = rsync_dir_to_vm(vm_ip, tokenizers_dirpath, self.remote_home)
        errors += check_subprocess_error(sc, ':Copy in:', '{} tokenizers copy in may have failed'.format(vm_name), send_mail=True, subject='{} tokenizers copy failed'.format(vm_name))

        return errors


class ImageSummary(ImageTask):
    def __init__(self):
        super().__init__()


class ImageClassification(ImageTask):
    def __init__(self):
        super().__init__()


class ImageObjectDetection(ImageTask):
    def __init__(self):
        super().__init__()


class ImageSegmentation(ImageTask):
    def __init__(self):
        super().__init__()


class NaturalLanguageProcessingSummary(NaturalLanguageProcessingTask):
    def __init__(self):
        super().__init__()


class NaturalLanguageProcessingSentiment(NaturalLanguageProcessingTask):
    def __init__(self):
        super().__init__()


class NaturalLanguageProcessingNamedEntityRecognition(NaturalLanguageProcessingTask):
    def __init__(self):
        super().__init__()


class NaturalLanguageProcessingQuestionAnswering(NaturalLanguageProcessingTask):
    def __init__(self):
        super().__init__()


