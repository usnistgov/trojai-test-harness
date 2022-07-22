import logging
import subprocess
from actor_executor.mail_io import TrojaiMail
from actor_executor import jsonschema_checker
from actor_executor.dataset import Dataset


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

# TODO: Implement
class Task(object):
    def __init__(self):
        pass

    def run_basic_checks(self, vm_ip, vm_name):
        errors = ''
        logging.info('Checking GPU status')
        sc = check_gpu(vm_ip)
        if sc != 0:
            msg = '"{}" GPU may be off-line with status code "{}".'.format(vm_name, sc)
            errors += ":GPU:"
            logging.error(msg)
            TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" GPU May be Offline'.format(vm_name), message=msg)

        return errors

    def run_container_checks(self, container_filepath):
        errors = ''
        logging.info('Checking for parameters in container')

        metaparameters_filepath = '/metaparameters.json'
        metaparameters_schema_filepath = '/metaparameters_schema.json'
        learned_parameters_dirpath = '/learned_parameters'

        sc = check_file_in_container(container_filepath, metaparameters_filepath)
        if sc != 0:
            logging.error('Metaparameters file "{}" not found in container'.format(metaparameters_filepath))
            errors += ':Container Parameters (metaparameters):'
        sc = check_file_in_container(container_filepath, metaparameters_schema_filepath)
        if sc != 0:
            logging.error('Metaparameters schema file "{}" not found in container.'.format(metaparameters_schema_filepath))
            errors += ':Container Parameters (metaparameters schema):'

        sc = check_dir_in_container(container_filepath, learned_parameters_dirpath)
        if sc != 0:
            logging.error('Learned parameters directory "{}" not found in container.'.format(learned_parameters_dirpath))
            errors += ':Container Parameters (learned parameters):'

        logging.info('Running checks on jsonschema')

        if not jsonschema_checker.is_container_configuration_valid(container_filepath):
            logging.error('Jsonschema contained errors.')
            errors += ':Container Parameters (jsonschema checker):'

        return errors

    def cleanup_vm(self, vm_ip, vm_name):
        errors = ''
        logging.info('Performing VM cleanup.')
        sc = cleanup_scratch(vm_ip)
        if sc != 0:
            logging.error('{} cleanup failed with status code {}'.format(vm_name, sc))
            errors += ':Cleanup:'

        return errors

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, data_split_name: str):
        raise NotImplementedError()
        # copy in evaluate scripts (all models and single model)
        # copy in submission filepath
        # update permissions of evaluate scripts
        # copy in round training dataset
        # copy in models
        # copy in tokenizers (NLP)
        # copy in source data


class ImageSummary(Task):
    def __init__(self):
        super().__init__()


class NaturalLanguageProcessingSummary(Task):
    def __init__(self):
        super().__init__()


class ImageClassification(Task):
    def __init__(self):
        super().__init__()


class ImageObjectDetection(Task):
    def __init__(self):
        super().__init__()


class ImageSegmentation(Task):
    def __init__(self):
        super().__init__()


class NaturalLanguageProcessingSentiment(Task):
    def __init__(self):
        super().__init__()


class NaturalLanguageProcessingNamedEntityRecognition(Task):
    def __init__(self):
        super().__init__()


class NaturalLanguageProcessingQuestionAnswering(Task):
    def __init__(self):
        super().__init__()


