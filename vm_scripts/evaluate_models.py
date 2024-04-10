
import argparse

from evaluate_task import *

VALID_SUBMISSION_TYPES = {'rl': EvaluateRLTask,
                          'nlp': EvaluateNLPTask,
                          'image': EvaluateImageTask,
                          'cyber': EvaluateCyberTask,
                          'cyber_pdf': EvaluateCyberPDFTask,
                          'clm': EvaluateClmTask}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point to execute containers')

    parser.add_argument('--models-dirpath',  type=str, description='The directory path to models to evaluate', required=True)
    parser.add_argument('--submission-type', type=str, choices=VALID_SUBMISSION_TYPES.keys(), description='The type of submission', required=True)
    parser.add_argument('--submission-filepath', type=str, description='The filepath to the submission', required=True)
    parser.add_argument('--home-dirpath', type=str, description='The directory path to home', required=True)
    parser.add_argument('--result-dirpath', type=str, description='The directory path for results', required=True)
    parser.add_argument('--scratch-dirpath', type=str, description='The directory path for scratch', required=True)
    parser.add_argument('--training-dataset-dirpath', type=str, description='The directory path to the training dataset', required=True)
    parser.add_argument('--metaparameter-filepath', type=str, description='The directory path for the metaparameters file when running custom metaparameters', required=False, default=None)
    parser.add_argument('--rsync-excludes', nargs='*', description='List of files to exclude for rsyncing data', required=False, default=None)
    parser.add_argument('--learned-parameters-dirpath', type=str, description='The directory path to the learned parameters', required=False, default=None)
    parser.add_argument('--source-dataset-dirpath', type=str, description='The source dataset directory path', required=False, default=None)
    parser.add_argument('--result-prefix-filename', type=str, description='The prefix name given to results', required=False, default=None)
    parser.add_argument('--subset-model-ids', nargs='*', description='List of model IDs to evaluate on', required=False, default=None)

    args, extras = parser.parse_known_args()

    models_dirpath = args.models_dirpath
    submission_type = args.submission_type
    submission_filepath = args.submission_filepath
    home_dirpath = args.home_dirpath
    result_diorpath = args.result_dirpath
    scratch_dirpath = args.scratch_dirpath
    training_dataset_dirpath = args.training_dataset_dirpath
    metaparameters_filepath = args.metaparameters_filepath
    source_dataset_dirpath = args.source_dataset_dirpath
    rsync_excludes = args.rsync_excludes
    result_prefix_filename = args.resule_prefix_filename
    subset_model_ids = args.subset_model_ids





