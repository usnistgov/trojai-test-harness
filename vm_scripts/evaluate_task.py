import argparse
import json
import sys
import os
import random
import shutil
import subprocess
import logging
import glob
import time

import signal
from spython.main import Client

from abc import ABC, abstractmethod


class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def rsync_dirpath(source_dirpath: str, dest_dirpath: str, rsync_args: list):
    params = ['rsync']
    params.extend(rsync_args)
    params.extend(glob.glob(source_dirpath))
    params.append(dest_dirpath)

    child = subprocess.Popen(params)
    return child.wait()

def clean_dirpath_contents(dirpath: str):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        except Exception as e:
            logging.info('Failed to delete {}, reason {}'.format(filepath, e))


class EvaluateTask(ABC):

    def __init__(self):
        pass

    def process_models(self):
        raise NotImplementedError('Must override execute_task')


    @abstractmethod
    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        raise NotImplementedError('Must override execute_task')


    @abstractmethod
    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        raise NotImplementedError('Must override execute_task')

class EvaluateTrojAITask(EvaluateTask):

    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):
        super().__init__()
        self.models_dirpath = models_dirpath
        self.submission_filepath = submission_filepath
        self.home_dirpath = home_dirpath
        self.result_dirpath = result_dirpath
        self.scratch_dirpath = scratch_dirpath
        self.training_dataset_dirpath = training_dataset_dirpath
        self.metaparameters_filepath = metaparameters_filepath
        self.rsync_excludes = rsync_excludes
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.source_dataset_dirpath = source_dataset_dirpath
        self.result_prefix_filename = result_prefix_filename
        self.subset_model_ids = subset_model_ids
        self.timeout_time = timeout_time
        self.total_execution_time = 0

        if not os.path.exists(self.result_dirpath):
            os.makedirs(self.result_dirpath, exist_ok=True)

        self.container_name = os.path.basename(self.submission_filepath)
        self.container_name = os.path.splitext(self.container_name)[0]

        if self.metaparameters_filepath is None:
            self.metaparameters_filepath = '/metaparameters.json'

        if self.rsync_excludes is None:
            self.rsync_excludes = []

        if self.learned_parameters_dirpath is None:
            self.learned_parameters_dirpath = '/learned_parameters'

        self.metaparameters_schema_filepath = '/metaparameters_schema.json'

        if self.subset_model_ids is None:
            self.subset_model_ids = []

        if self.result_prefix_filename is None:
            self.result_prefix_filename = ''

        # std:out from the Client.run(container_instance, container_args, return_result=True)
        # will be directed to this log file
        log_filepath = os.path.join(self.result_dirpath, '{}.out'.format(self.container_name))
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(message)s",
                            filename=log_filepath)
        # capture container stdout and stderror
        logger = logging.getLogger()
        # THIS CANNOT CO-EXIST WITH A logging.StreamHandler() AS IT WILL CAUSE AN INFINITE LOGGING LOOP
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

    def execute_container(self, container_instance, active_dirpath, container_scratch_dirpath, model_dirname):

        active_result_filepath = os.path.join(active_dirpath, 'result.txt')

        container_args = self.get_execute_task_args(active_dirpath, container_scratch_dirpath, active_result_filepath)

        remaining_time = self.timeout_time - self.total_execution_time

        if remaining_time <= 0:
            raise TimeoutError('Timeout')

        execute_start = time.time()
        with timeout(seconds=int(remaining_time)):
            result = Client.run(container_instance, container_args, return_result=True)

        execute_end = time.time()
        self.total_execution_time = self.total_execution_time + (execute_end - execute_start)

        # copy results back to real output filename
        if os.path.exists(active_result_filepath):
            shutil.copy(active_result_filepath, os.path.join(self.result_dirpath, '{}{}.txt'.format(self.result_prefix_filename, model_dirname)))

        return result

    def process_models(self):
        active_dirpath = os.path.join(self.scratch_dirpath, 'active')
        container_scratch_dirpath = os.path.join(self.scratch_dirpath, 'container-scratch')

        if not os.path.exists(active_dirpath):
            os.makedirs(active_dirpath)

        if not os.path.exists(container_scratch_dirpath):
            os.makedirs(container_scratch_dirpath)

        model_files = [fn for fn in os.listdir(self.models_dirpath) if fn.startswith('id-')]
        random.shuffle(model_files)

        options = self.get_singularity_instance_options(active_dirpath, container_scratch_dirpath)
        logging.info("Starting container instance.")
        container_instance = Client.instance(self.submission_filepath, options=options)
        logging.info("Container started.")
        for model_idx in range(len(model_files)):
            model_dirname = model_files[model_idx]

            # Clean up scratch and active dir prior to running
            clean_dirpath_contents(container_scratch_dirpath)
            clean_dirpath_contents(active_dirpath)

            model_dirpath = os.path.join(self.models_dirpath, model_dirname)
            rsync_params = ['-ar', '--prune-empty-dirs', '--delete']

            for rsync_exclude in self.rsync_excludes:
                rsync_params.append('--exclude={}'.format(rsync_exclude))

            rsync_dirpath(os.path.join(model_dirpath, '*'), active_dirpath, rsync_params)

            # check for reduced-config, and copy it if it does exist to config.json
            reduced_config_filepath = os.path.join(active_dirpath, 'reduced-config.json')
            if os.path.exists(reduced_config_filepath):
                shutil.copy(reduced_config_filepath, os.path.join(active_dirpath, 'config.json'))

            logging.info('Starting execution of {} ({}/{})'.format(model_dirname, model_idx, len(model_files)))

            container_start_time = time.time()

            result = self.execute_container(container_instance, active_dirpath, container_scratch_dirpath, model_dirname)

            return_code = -1
            if 'return_code' in result:
                return_code = result['return_code']
            else:
                logging.error('Failed to obtain result from singularity execution: {}'.format(result))

            container_end_time = time.time()

            container_exec_time = container_end_time - container_start_time



            with open(os.path.join(self.result_dirpath, '{}{}-walltime.txt'.format(self.result_prefix_filename, model_dirname)), 'w') as f:
                f.write('{}'.format(container_exec_time))

            logging.info('Finished executing {}, returned status code: {}'.format(model_dirname, return_code))



        logging.info("All model executions complete, stopping continer.")
        container_instance.stop()
        logging.info("Container stopped.")

    @abstractmethod
    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        gpu_option = ''
        if uses_gpu:
            gpu_option = '--nv'

        options = ['--contain', '--bind', active_dirpath, '--bind', scratch_dirpath]

        if self.training_dataset_dirpath is not None:
            options.extend(['--bind', '{}:{}:ro'.format(self.training_dataset_dirpath, self.training_dataset_dirpath)])

        options.append(gpu_option)

        if self.source_dataset_dirpath is not None:
            options.extend(['--bind', '{}:{}:ro'.format(self.source_dataset_dirpath, self.source_dataset_dirpath)])
        return options

    @abstractmethod
    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        raise NotImplementedError('Must override execute_task')



class EvaluateImageTask(EvaluateTrojAITask):
    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath]

        if self.training_dataset_dirpath is not None:
            args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args


class EvaluateClmTask(EvaluateTrojAITask):
    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        options = super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)
        return options

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        args = ['--model_filepath', active_dirpath,
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath]

        if self.training_dataset_dirpath is not None:
            args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args

class EvaluateClmInstructTask(EvaluateTrojAITask):
    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)


    def process_models(self):
        os.environ['HF_HUB_OFFLINE'] = str(1)

        active_dirpath = os.path.join(self.scratch_dirpath, 'active')
        container_scratch_dirpath = os.path.join(self.scratch_dirpath, 'container-scratch')

        if not os.path.exists(active_dirpath):
            os.makedirs(active_dirpath)

        if not os.path.exists(container_scratch_dirpath):
            os.makedirs(container_scratch_dirpath)

        model_files = [fn for fn in os.listdir(self.models_dirpath) if fn.startswith('id-')]
        random.shuffle(model_files)

        options = self.get_singularity_instance_options(active_dirpath, container_scratch_dirpath)
        options.extend(['--bind', '/home/trojai/.cache/huggingface/hub'])
        logging.info("Starting container instance.")
        container_instance = Client.instance(self.submission_filepath, options=options)
        logging.info("Container started with options {}.".format(options))
        for model_idx in range(len(model_files)):
            model_dirname = model_files[model_idx]

            # Clean up scratch and active dir prior to running
            clean_dirpath_contents(container_scratch_dirpath)
            clean_dirpath_contents(active_dirpath)

            model_dirpath = os.path.join(self.models_dirpath, model_dirname)
            rsync_params = ['-ar', '--prune-empty-dirs', '--delete']

            for rsync_exclude in self.rsync_excludes:
                rsync_params.append('--exclude={}'.format(rsync_exclude))

            rsync_dirpath(os.path.join(model_dirpath, '*'), active_dirpath, rsync_params)

            # check for reduced-config, and copy it if it does exist to config.json
            reduced_config_filepath = os.path.join(active_dirpath, 'reduced-config.json')
            if os.path.exists(reduced_config_filepath):
                shutil.copy(reduced_config_filepath, os.path.join(active_dirpath, 'round_config.json'))

            logging.info('Starting execution of {} ({}/{})'.format(model_dirname, model_idx, len(model_files)))

            container_start_time = time.time()

            result = self.execute_container(container_instance, active_dirpath, container_scratch_dirpath, model_dirname)

            return_code = -1
            if 'return_code' in result:
                return_code = result['return_code']
            else:
                logging.error('Failed to obtain result from singularity execution: {}'.format(result))

            container_end_time = time.time()

            container_exec_time = container_end_time - container_start_time



            with open(os.path.join(self.result_dirpath, '{}{}-walltime.txt'.format(self.result_prefix_filename, model_dirname)), 'w') as f:
                f.write('{}'.format(container_exec_time))

            logging.info('Finished executing {}, returned status code: {}'.format(model_dirname, return_code))



        logging.info("All model executions complete, stopping continer.")
        container_instance.stop()
        logging.info("Container stopped.")


    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        options = super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)
        return options

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        args = ['--model_filepath', active_dirpath,
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath]

        if self.training_dataset_dirpath is not None:
            args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args

class EvaluateNLPTask(EvaluateTrojAITask):

    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)

        parser = argparse.ArgumentParser(description='Parser for NLP')
        parser.add_argument('--tokenizer-dirpath', type=str, help='The directory path to tokenizers', required=True)

        args, extras = parser.parse_known_args()

        self.tokenizer_dirpath = args.tokenizer_dirpath

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        options = super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)
        options.extend(['--bind', '{}:{}:ro'.format(self.tokenizer_dirpath, self.tokenizer_dirpath)])
        return options

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        # Load model architecture to get tokenizer filepath
        config_filepath = os.path.join(active_dirpath, 'config.json')
        model_architecture = 'unknown'
        with open(config_filepath) as f:
            config_dict = json.load(f)
            if 'model_architecture' in config_dict:
                model_architecture = config_dict['model_architecture']

        tokenizer_filepath = os.path.join(self.tokenizer_dirpath, '{}.pt'.format(model_architecture.replace('/','-')))

        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath,
                '--tokenizer_filepath', tokenizer_filepath]

        if self.training_dataset_dirpath is not None:
            args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args



class EvaluateRLTask(EvaluateTrojAITask):

    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath]

        if self.training_dataset_dirpath is not None:
            args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args


class EvaluateRLColorfulMemoryTask(EvaluateRLTask):

    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=False):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath]

        if self.training_dataset_dirpath is not None:
            args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args

class EvaluateRLSafetyGym(EvaluateRLColorfulMemoryTask):
    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)


class EvaluateCyberTask(EvaluateTrojAITask):
    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)


    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):

        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath]

        if self.training_dataset_dirpath is not None:
            args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args

class EvaluateCyberPDFTask(EvaluateTrojAITask):

    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):

        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)

        parser = argparse.ArgumentParser(description='Parser for Cyber')
        parser.add_argument('--scale-params-filepath', type=str, help='The filepath to the scale parameters file', required=True)

        args, extras = parser.parse_known_args()

        self.scale_params_filepath = args.scale_params_filepath
        self.scale_params_filename = os.path.basename(self.scale_params_filepath)

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        active_scale_params_filepath = os.path.join(active_dirpath, self.scale_params_filename)
        shutil.copy(self.scale_params_filepath, active_scale_params_filepath)

        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath,
                '--scale_parameters_filepath', active_scale_params_filepath]

        if self.training_dataset_dirpath is not None:
            args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args

class EvaluateMitigationTask(EvaluateTrojAITask):
    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):
        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)


    def execute_container(self, container_instance, active_dirpath, container_scratch_dirpath, model_dirname):

        remaining_time = self.timeout_time - self.total_execution_time

        if remaining_time <= 0:
            raise TimeoutError('Timeout')

        execute_start = time.time()

        with timeout(seconds=int(remaining_time)):
            # Called for each model.
            # Setup mitigation Arguments
            model_filepath = os.path.join(active_dirpath, 'model.pt')
            output_dirpath = os.path.join(active_dirpath, 'output')
            output_model_name = 'mitigated.pt'

            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)

            mitigate_dataset_dirpath = os.path.join(active_dirpath, 'mitigate-example-data')

            mitigate_args = ['mitigate',
                             '--model_filepath', model_filepath,
                             '--metaparameters_filepath', self.metaparameters_filepath,
                             '--schema_filepath', self.metaparameters_schema_filepath,
                             '--dataset', mitigate_dataset_dirpath,
                             '--scratch_dirpath', container_scratch_dirpath,
                             '--output_dirpath', output_dirpath,
                             '--model_output_name', output_model_name]

            if self.training_dataset_dirpath is not None:
                mitigate_args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

            # Execute mitigate
            result = Client.run(container_instance, mitigate_args, return_result=True)
            logging.info('Output from mitigate step: {}'.format(result))

            # Setup test arguments
            mitigated_model_filepath = os.path.join(output_dirpath, output_model_name)

            if not os.path.exists(mitigated_model_filepath):
                logging.error('Failed to find mitigated model, mitigate step may have errors, using original model.')
                mitigated_model_filepath = model_filepath

            test_dataset_dirpath = os.path.join(active_dirpath, 'test-example-data')

            test_args = ['test',
                         '--metaparameters_filepath', self.metaparameters_filepath,
                         '--schema_filepath', self.metaparameters_schema_filepath,
                         '--model_filepath', mitigated_model_filepath,
                         '--dataset', test_dataset_dirpath,
                         '--scratch_dirpath', container_scratch_dirpath,
                         '--output_dirpath', output_dirpath,
                         ]

            if self.training_dataset_dirpath is not None:
                test_args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

            result = Client.run(container_instance, test_args, return_result=True)

        output_results_filepath = os.path.join(output_dirpath, 'results.json')

        # copy results back to real output filename
        if os.path.exists(output_results_filepath):
            shutil.copy(output_results_filepath, os.path.join(self.result_dirpath, '{}{}.json'.format(self.result_prefix_filename, model_dirname)))
        else:
            logging.warning('{} is missing the output result file {}'.format(model_dirname, output_results_filepath))

        execute_end = time.time()

        self.total_execution_time = self.total_execution_time + (execute_end - execute_start)

        return result

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        args = []

        return args


class EvaluateLLMMitigationTask(EvaluateTrojAITask):
    def __init__(self, models_dirpath: str,
                 submission_filepath: str,
                 home_dirpath: str,
                 result_dirpath: str,
                 scratch_dirpath: str,
                 training_dataset_dirpath: str,
                 metaparameters_filepath: str,
                 rsync_excludes: list,
                 learned_parameters_dirpath: str,
                 source_dataset_dirpath: str,
                 result_prefix_filename: str,
                 subset_model_ids: list,
                 timeout_time: int):
        super().__init__(models_dirpath=models_dirpath,
                         submission_filepath=submission_filepath,
                         home_dirpath=home_dirpath,
                         result_dirpath=result_dirpath,
                         scratch_dirpath=scratch_dirpath,
                         training_dataset_dirpath=training_dataset_dirpath,
                         metaparameters_filepath=metaparameters_filepath,
                         rsync_excludes=rsync_excludes,
                         learned_parameters_dirpath=learned_parameters_dirpath,
                         source_dataset_dirpath=source_dataset_dirpath,
                         result_prefix_filename=result_prefix_filename,
                         subset_model_ids=subset_model_ids,
                         timeout_time=timeout_time)



    def process_models(self):
        active_dirpath = os.path.join(self.scratch_dirpath, 'active')
        container_scratch_dirpath = os.path.join(self.scratch_dirpath, 'container-scratch')

        if not os.path.exists(active_dirpath):
            os.makedirs(active_dirpath)

        if not os.path.exists(container_scratch_dirpath):
            os.makedirs(container_scratch_dirpath)

        model_files = [fn for fn in os.listdir(self.models_dirpath) if fn.startswith('id-')]
        random.shuffle(model_files)

        options = self.get_singularity_instance_options(active_dirpath, container_scratch_dirpath)
        logging.info("Starting container instance.")
        logging.info('Binding performer container with options: {}'.format(options))
        container_instance = Client.instance(self.submission_filepath, options=options)
        logging.info("Container started.")

        logging.info('Starting eval container instance.')
        eval_options = self.get_singularity_instance_options(active_dirpath, container_scratch_dirpath)
        eval_options.extend(['--bind', self.models_dirpath, '--bind', os.path.join(self.home_dirpath, '.cache', 'huggingface')])

        logging.info('Binding mitigation evaluator with options: {}'.format(eval_options))

        eval_instance = Client.instance(os.path.join(self.home_dirpath, 'mitigation_evaluator.sif'), options=eval_options)
        logging.info('Eval container started.')

        # Add test-example-data to rsync excludes
        self.rsync_excludes.append('test-example-data')

        for model_idx in range(len(model_files)):
            model_dirname = model_files[model_idx]

            # Clean up scratch and active dir prior to running
            clean_dirpath_contents(container_scratch_dirpath)
            clean_dirpath_contents(active_dirpath)

            model_dirpath = os.path.join(self.models_dirpath, model_dirname)
            rsync_params = ['-ar', '--prune-empty-dirs', '--delete']


            for rsync_exclude in self.rsync_excludes:
                rsync_params.append('--exclude={}'.format(rsync_exclude))

            rsync_dirpath(os.path.join(model_dirpath, '*'), active_dirpath, rsync_params)

            # check for reduced-config, and copy it if it does exist to config.json
            reduced_config_filepath = os.path.join(active_dirpath, 'reduced-config.json')
            if os.path.exists(reduced_config_filepath):
                shutil.copy(reduced_config_filepath, os.path.join(active_dirpath, 'round_config.json'))

            logging.info('Starting execution of {} ({}/{})'.format(model_dirname, model_idx, len(model_files)))

            container_start_time = time.time()

            result = self.execute_container_llm(container_instance, eval_instance, active_dirpath, container_scratch_dirpath, model_dirname)

            return_code = -1
            if 'return_code' in result:
                return_code = result['return_code']
            else:
                logging.error('Failed to obtain result from singularity execution: {}'.format(result))

            container_end_time = time.time()

            container_exec_time = container_end_time - container_start_time



            with open(os.path.join(self.result_dirpath, '{}{}-walltime.txt'.format(self.result_prefix_filename, model_dirname)), 'w') as f:
                f.write('{}'.format(container_exec_time))

            logging.info('Finished executing {}, returned status code: {}'.format(model_dirname, return_code))



        logging.info("All model executions complete, stopping continer.")
        container_instance.stop()
        eval_instance.stop()
        logging.info("Container stopped.")


    def execute_container_llm(self, container_instance, eval_instance, active_dirpath, container_scratch_dirpath, model_dirname):

        remaining_time = self.timeout_time - self.total_execution_time

        if remaining_time <= 0:
            raise TimeoutError('Timeout')

        with timeout(seconds=int(remaining_time)):
            # Called for each model.
            # Setup mitigation Arguments
            model_dirpath = active_dirpath
            output_dirpath = os.path.join(active_dirpath, 'output')

            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)

            # mitigate_dataset_dirpath = os.path.join(active_dirpath, 'mitigate-example-data')

            mitigate_args = ['mitigate',
                             '--metaparameters_filepath', self.metaparameters_filepath,
                             '--schema_filepath', self.metaparameters_schema_filepath,
                             '--model', model_dirpath,
                             '--output_dirpath', output_dirpath,
                             '--scratch_dirpath', container_scratch_dirpath,
                             '--batch_size', str(1)]

            if self.training_dataset_dirpath is not None:
                mitigate_args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

            logging.info('Running performer container with args: {}'.format(mitigate_args))
            execute_start = time.time()

            # Execute mitigate
            result = Client.run(container_instance, mitigate_args, return_result=True)
            execute_end = time.time()

            logging.info('Output from mitigate step: {}'.format(result))

            # Setup test arguments
            checkpoint_filepath = output_dirpath
            round_config_filepath = os.path.join(self.models_dirpath, model_dirname, 'round_config.json')
            tokenizer_config_filepath = os.path.join(output_dirpath, 'tokenizer_config.json')
            num_samples = 100
            test_dataset_filepath = os.path.join(self.models_dirpath, model_dirname, 'test-example-data', 'prompts.json')
            output_filepath = os.path.join(output_dirpath, 'results.json')

            test_args = [checkpoint_filepath,
                         round_config_filepath,
                         tokenizer_config_filepath,
                         test_dataset_filepath,
                         str(num_samples),
                         output_filepath
                         ]
            logging.info('Running evaluate with args: {}'.format(test_args))
            # if self.training_dataset_dirpath is not None:
            #     test_args.extend(['--round_training_dataset_dirpath', self.training_dataset_dirpath])

            result = Client.run(eval_instance, test_args, return_result=True)

        output_results_filepath = os.path.join(output_dirpath, 'results.json')

        # copy results back to real output filename
        if os.path.exists(output_results_filepath):
            shutil.copy(output_results_filepath, os.path.join(self.result_dirpath, '{}{}.json'.format(self.result_prefix_filename, model_dirname)))
        else:
            logging.warning('{} is missing the output result file {}'.format(model_dirname, output_results_filepath))


        self.total_execution_time = self.total_execution_time + (execute_end - execute_start)

        return result

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath, uses_gpu)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        # active_scale_params_filepath = os.path.join(active_dirpath, self.scale_params_filename)
        # shutil.copy(self.scale_params_filepath, active_scale_params_filepath)

        args = []

        return args


if __name__ == '__main__':
    VALID_TASK_TYPES = {
      'rl': EvaluateRLTask,
      'rl_color': EvaluateRLColorfulMemoryTask,
      'nlp': EvaluateNLPTask,
      'image': EvaluateImageTask,
      'cyber': EvaluateCyberTask,
      'cyber_pdf': EvaluateCyberPDFTask,
      'clm': EvaluateClmTask,
      'clm_inst': EvaluateClmInstructTask,
      'image_classification_mitigation': EvaluateMitigationTask,
      'llm_mitigation': EvaluateLLMMitigationTask,
      'rl_gym': EvaluateRLSafetyGym
    }

    parser = argparse.ArgumentParser(description='Entry point to execute containers')

    parser.add_argument('--timeout', type=int, help='The amount of time to timeout the execution in seconds')
    parser.add_argument('--models-dirpath',  type=str, help='The directory path to models to evaluate', required=True)
    parser.add_argument('--task-type', type=str, choices=VALID_TASK_TYPES.keys(), help='The type of submission', required=True)
    parser.add_argument('--submission-filepath', type=str, help='The filepath to the submission', required=True)
    parser.add_argument('--home-dirpath', type=str, help='The directory path to home', required=True)
    parser.add_argument('--result-dirpath', type=str, help='The directory path for results', required=True)
    parser.add_argument('--scratch-dirpath', type=str, help='The directory path for scratch', required=True)
    parser.add_argument('--training-dataset-dirpath', type=str, help='The directory path to the training dataset', required=False, default=None)
    parser.add_argument('--metaparameter-filepath', type=str, help='The directory path for the metaparameters file when running custom metaparameters', required=False, default=None)
    parser.add_argument('--rsync-excludes', nargs='*', help='List of files to exclude for rsyncing data', required=False, default=None)
    parser.add_argument('--learned-parameters-dirpath', type=str, help='The directory path to the learned parameters', required=False, default=None)
    parser.add_argument('--source-dataset-dirpath', type=str, help='The source dataset directory path', required=False, default=None)
    parser.add_argument('--result-prefix-filename', type=str, help='The prefix name given to results', required=False, default=None)
    parser.add_argument('--subset-model-ids', nargs='*', help='List of model IDs to evaluate on', required=False, default=None)

    args, extras = parser.parse_known_args()

    task_type = args.task_type

    evaluate_task_instance = VALID_TASK_TYPES[task_type](models_dirpath=args.models_dirpath,
                                                         submission_filepath=args.submission_filepath,
                                                         home_dirpath=args.home_dirpath,
                                                         result_dirpath=args.result_dirpath,
                                                         scratch_dirpath=args.scratch_dirpath,
                                                         training_dataset_dirpath=args.training_dataset_dirpath,
                                                         metaparameters_filepath=args.metaparameter_filepath,
                                                         rsync_excludes=args.rsync_excludes,
                                                         learned_parameters_dirpath=args.learned_parameters_dirpath,
                                                         source_dataset_dirpath=args.source_dataset_dirpath,
                                                         result_prefix_filename=args.result_prefix_filename,
                                                         subset_model_ids=args.subset_model_ids,
                                                         timeout_time=args.timeout)

    try:
        evaluate_task_instance.process_models()
    except TimeoutError as e:
        logging.info('Your submission has failed to process all models within the timelimit ({}s)'.format(args.timeout))
        exit(-9)
