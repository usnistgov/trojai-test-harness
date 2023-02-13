import argparse
import json
import os
import random
import shutil
import subprocess
import logging
from logging import FileHandler
import time
from spython.main import Client

from abc import ABC, abstractmethod

def rsync_dirpath(source_dirpath: str, dest_dirpath: str, rsync_args: list, with_shell=False):
    params = ['rsync']
    params.extend(rsync_args)
    params.append(source_dirpath)
    params.append(dest_dirpath)

    logging.info(params)
    child = subprocess.Popen(params, shell=with_shell)
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
                 subset_model_ids: list):
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


        self.container_name = os.path.splitext(self.submission_filepath)[0]

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

        handler = FileHandler(os.path.join(self.result_dirpath, '{}.out'.format(self.container_name)))

        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                            handlers=[handler])

    def process_models(self):
        active_dirpath = os.path.join(self.scratch_dirpath, 'active')
        container_scratch_dirpath = os.path.join(self.scratch_dirpath, 'container-scratch')

        if not os.path.exists(active_dirpath):
            os.makedirs(active_dirpath)

        if not os.path.exists(container_scratch_dirpath):
            os.makedirs(container_scratch_dirpath)

        model_files = os.listdir(self.models_dirpath)
        random.shuffle(model_files)

        options = self.get_singularity_instance_options(active_dirpath, container_scratch_dirpath)
        container_instance = Client.instance(self.submission_filepath, options=options)
        for model_dirname in model_files:
            if not model_dirname.startswith('id'):
                continue

            # Clean up scratch and active dir prior to running
            clean_dirpath_contents(container_scratch_dirpath)
            clean_dirpath_contents(active_dirpath)

            model_dirpath = os.path.join(self.models_dirpath, model_dirname)
            rsync_params = ['-ar', '--prune-empty-dirs', '--delete']

            for rsync_exclude in self.rsync_excludes:
                rsync_params.append('--exclude={}'.format(rsync_exclude))

            rsync_dirpath(os.path.join(model_dirpath, '*'), active_dirpath, rsync_params, with_shell=True)

            # check for reduced-config, and copy it if it does exist to config.json
            reduced_config_filepath = os.path.join(active_dirpath, 'reduced-config.json')
            if os.path.exists(reduced_config_filepath):
                shutil.copy(reduced_config_filepath, os.path.join(active_dirpath, 'config.json'))

            logging.info('Starting execution of {}'.format(model_dirname))

            active_result_filepath = os.path.join(active_dirpath, 'result.txt')
            container_output_filepath = os.path.join(self.result_dirpath, '{}.out'.format(self.container_name))

            container_start_time = time.time()
            container_args = self.get_execute_task_args(active_dirpath, container_scratch_dirpath, active_result_filepath)

            result = Client.run(container_instance, container_args, return_result=True)

            if 'message' in result:
                with open(container_output_filepath, 'w') as f:
                    f.write(result['message'])
            else:
                logging.error('Failed to obtain result from singularity execution: {}'.format(result))

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

            # copy results back to real output filename
            shutil.copy(active_result_filepath, os.path.join(self.result_dirpath, '{}{}.txt'.format(self.result_prefix_filename, model_dirname)))

    @abstractmethod
    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath):
        options = ['--contain', '--bind', active_dirpath, '--bind', scratch_dirpath, '--bind', '{}:{}:ro'.format(self.training_dataset_dirpath, self.training_dataset_dirpath), '--nv']
        if self.source_dataset_dirpath is not None:
            options.extend(['--bind', '{}:{}:ro'.format(self.source_dataset_dirpath, self.source_dataset_dirpath)])
        return options

    @abstractmethod
    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        raise NotImplementedError('Must override execute_task')


class EvaluateImageTask(EvaluateTask):
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
                 subset_model_ids: list):
        super().__init__(models_dirpath, submission_filepath, home_dirpath, result_dirpath, scratch_dirpath,
                         training_dataset_dirpath, metaparameters_filepath, rsync_excludes, learned_parameters_dirpath,
                         source_dataset_dirpath, result_prefix_filename, subset_model_ids)

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--round_training_dataset_dirpath', self.training_dataset_dirpath,
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath]

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args


class EvaluateNLPTask(EvaluateTask):

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
                 subset_model_ids: list):
        super().__init__(models_dirpath, submission_filepath, home_dirpath, result_dirpath, scratch_dirpath,
                         training_dataset_dirpath, metaparameters_filepath, rsync_excludes, learned_parameters_dirpath,
                         source_dataset_dirpath, result_prefix_filename, subset_model_ids)

        parser = argparse.ArgumentParser(description='Parser for NLP')
        parser.add_argument('--tokenizer-dirpath', type=str, help='The directory path to tokenizers', required=True)

        args, extras = parser.parse_known_args()

        self.tokenizer_dirpath = args.tokenizer_dirpath

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath):
        options = super().get_singularity_instance_options(active_dirpath, scratch_dirpath)
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

        tokenizer_filepath = os.path.join(self.tokenizer_dirpath, '{}.pt'.format(model_architecture))

        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--round_training_dataset_dirpath', self.training_dataset_dirpath,
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath,
                '--tokenizer_filepath', tokenizer_filepath]

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args



class EvaluateRLTask(EvaluateTask):

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
                 subset_model_ids: list):
        super().__init__(models_dirpath, submission_filepath, home_dirpath, result_dirpath, scratch_dirpath,
                         training_dataset_dirpath, metaparameters_filepath, rsync_excludes, learned_parameters_dirpath,
                         source_dataset_dirpath, result_prefix_filename, subset_model_ids)
    # TODO: Implement


class EvaluateCyberTask(EvaluateTask):

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
                 subset_model_ids: list):
        super().__init__(models_dirpath, submission_filepath, home_dirpath, result_dirpath, scratch_dirpath,
                         training_dataset_dirpath, metaparameters_filepath, rsync_excludes, learned_parameters_dirpath,
                         source_dataset_dirpath, result_prefix_filename, subset_model_ids)

        parser = argparse.ArgumentParser(description='Parser for Cyber')
        parser.add_argument('--scale-params-filepath', type=str, help='The filepath to the scale parameters file', required=True)

        args, extras = parser.parse_known_args()

        self.scale_params_filepath = args.scale_params_filepath
        self.scale_params_filename = os.path.basename(self.scale_params_filepath)

    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath):
        return super().get_singularity_instance_options(active_dirpath, scratch_dirpath)

    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        active_scale_params_filepath = os.path.join(active_dirpath, self.scale_params_filename)
        shutil.copy(self.scale_params_filepath, active_scale_params_filepath)



        args = ['--model_filepath', os.path.join(active_dirpath, 'model.pt'),
                '--result_filepath', active_result_filepath,
                '--scratch_dirpath', container_scratch_dirpath,
                '--examples_dirpath', os.path.join(active_dirpath, 'clean-example-data'),
                '--round_training_dataset_dirpath', self.training_dataset_dirpath,
                '--metaparameters_filepath', self.metaparameters_filepath,
                '--schema_filepath', self.metaparameters_schema_filepath,
                '--learned_parameters_dirpath', self.learned_parameters_dirpath,
                '--scale_parameters_filepath', active_scale_params_filepath]

        if self.source_dataset_dirpath is not None:
            args.extend(['--source_dataset_dirpath', self.source_dataset_dirpath])

        return args

if __name__ == '__main__':
    VALID_TASK_TYPES = {'rl': EvaluateRLTask,
                              'nlp': EvaluateNLPTask,
                              'image': EvaluateImageTask,
                              'cyber': EvaluateCyberTask}

    parser = argparse.ArgumentParser(description='Entry point to execute containers')

    parser.add_argument('--models-dirpath',  type=str, help='The directory path to models to evaluate', required=True)
    parser.add_argument('--task-type', type=str, choices=VALID_TASK_TYPES.keys(), help='The type of submission', required=True)
    parser.add_argument('--submission-filepath', type=str, help='The filepath to the submission', required=True)
    parser.add_argument('--home-dirpath', type=str, help='The directory path to home', required=True)
    parser.add_argument('--result-dirpath', type=str, help='The directory path for results', required=True)
    parser.add_argument('--scratch-dirpath', type=str, help='The directory path for scratch', required=True)
    parser.add_argument('--training-dataset-dirpath', type=str, help='The directory path to the training dataset', required=True)
    parser.add_argument('--metaparameter-filepath', type=str, help='The directory path for the metaparameters file when running custom metaparameters', required=False, default=None)
    parser.add_argument('--rsync-excludes', nargs='*', help='List of files to exclude for rsyncing data', required=False, default=None)
    parser.add_argument('--learned-parameters-dirpath', type=str, help='The directory path to the learned parameters', required=False, default=None)
    parser.add_argument('--source-dataset-dirpath', type=str, help='The source dataset directory path', required=False, default=None)
    parser.add_argument('--result-prefix-filename', type=str, help='The prefix name given to results', required=False, default=None)
    parser.add_argument('--subset-model-ids', nargs='*', help='List of model IDs to evaluate on', required=False, default=None)

    args, extras = parser.parse_known_args()

    task_type = args.task_type

    evaluate_task_instance = VALID_TASK_TYPES[task_type](args.models_dirpath, args.submission_filepath, args.home_dirpath,
                         args.result_dirpath, args.scratch_dirpath, args.training_dataset_dirpath, args.metaparameter_filepath,
                         args.rsync_excludes, args.learned_parameters_dirpath, args.source_dataset_dirpath,
                         args.result_prefix_filename, args.subset_model_ids)

    evaluate_task_instance.process_models()
