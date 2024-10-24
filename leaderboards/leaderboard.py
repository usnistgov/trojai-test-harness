# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import collections
import copy
import datetime
import json
import os
import typing

import jsonpickle
import pandas as pd
import numpy as np
from airium import Airium
from typing import Union, Tuple, Dict, Any

from leaderboards.drive_io import DriveIO
from leaderboards.python_utils import update_object_values, get_value
from leaderboards.trojai_config import TrojaiConfig
from leaderboards import json_io
from leaderboards.dataset import DatasetManager, Dataset
from leaderboards.metrics import *
from leaderboards.tasks import *
from leaderboards.summary_metrics import *
from leaderboards.results_manager import ResultsManager


class Leaderboard(object):
    LEADERBOARD_TYPES = ['trojai', 'mitigation', 'llm_mitigation']

    ALL_TASK_NAMES = {'image_summary': ImageSummary,
                        'nlp_summary': NaturalLanguageProcessingSummary,
                        'image_classification': ImageClassification,
                        'image_object_detection' : ImageObjectDetection,
                        'image_segmentation': ImageSegmentation,
                        'nlp_sentiment': NaturalLanguageProcessingSentiment,
                        'nlp_ner': NaturalLanguageProcessingNamedEntityRecognition,
                        'nlp_qa': NaturalLanguageProcessingQuestionAnswering,
                        "cyber": CyberTask,
                        "cyber_windows_pe_malware": CyberWindowsPEMalware,
                        "cyber_apk_malware": CyberApkMalware,
                        "cyber_pdf_malware": CyberPdfMalware,
                        "rl_lavaworld": ReinforcementLearningLavaWorld,
                        "rl_colorful": ReinforcementLearningColorfulMemory,
                        "rl_gym": ReinforcementLearningSafetyGym,
                        'causal_language': CausalLanguageModeling,
                        'image_classification_mitigation': ImageClassificationMitigationTask,
                        'llm_mitigation': LLMMitigationTask
                      }

    ALL_METRIC_NAMES = {
        'AverageCrossEntropy': AverageCrossEntropy,
        'GroupedCrossEntropyViolin': GroupedCrossEntropyViolin,
        'GroupedROC-AUC': Grouped_ROC_AUC,
        'CrossEntropyConfidenceInterval': CrossEntropyConfidenceInterval,
        'BrierScore': BrierScore,
        'ROC_AUC': ROC_AUC,
        'DEX_Factor_csv': DEX_Factor_csv,
        'Mit_Psn_Acc': MitigationPoisonedAccuracyOnPoisonedModel,
        'Mit_Clean_Acc_Psn': MitigationCleanAccuracyOnPoisonedModel,
        'Mit_Clean_acc': MitigationAccuracyOnCleanModel,
        'Mit_Overall_acc': MitigationAccuraccyOverall,
        'Mit_fidelity': MitigationFidelityMetric
    }

    ALL_SUMMARY_METRIC_NAMES = {
        'SummaryAverageCEOverTime': SummaryAverageCEOverTime
    }

    GENERAL_SLURM_QUEUE_NAME = 'es'
    STS_SLURM_QUEUE_NAME = 'sts'
    INFO_FILENAME = 'info.json'

    SLURM_QUEUE_NAMES = [GENERAL_SLURM_QUEUE_NAME, STS_SLURM_QUEUE_NAME]

    TABLE_NAMES = ['results', 'all-results', 'jobs']

    DEFAULT_SCHEMA_SUMMARY_SUFFIX = 'schema_summary.csv'

    # This is a running list of unique dataset names to show tooltip text
    DATASET_DESCRIPTIONS = {
        'train': 'train: The train dataset that is distributed with each round.',
        'test': 'test: The test dataset that is sequestered/hidden, used for evaluation. Submissions here should be fully realized with complete schema and parameters.',
        'sts': 'sts: The sts dataset uses a subset of the train dataset, useful for debugging container submission.',
        'dev': 'dev: The dev dataset uses the test dataset, and should be used for in-development solutions. Schemas must be valid, but do not need to be complete. Results do not count towards the program.',
        'holdout': 'holdout: The holdout dataset that is sequestered/hidden, used for holdout evaluation.'
    }
    def __init__(self, name: str, task_name: str, trojai_config: TrojaiConfig):
        if '_' in name:
            raise RuntimeError('Invalid leaderboard name: {}, should not have any underscores "_"'.format(name))
        self.name = name
        self.task_name = task_name
        self.revision = 1
        self.check_for_missing_metrics = True

        self.submission_dirpath = os.path.join(trojai_config.submission_dirpath, self.name)
        self.submissions_filepath = os.path.join(self.submission_dirpath, 'submissions.json')

        self.leaderboard_results_filepath = os.path.join(trojai_config.leaderboard_results_dirpath, '{}_results.parquet'.format(self.name))

        self.highlight_old_submissions = False
        self.html_leaderboard_priority = 0
        self.html_data_split_name_priorities = {}
        self.html_table_sort_options = {}

        self.evaluation_metric_name = None
        self.summary_metrics = []
        self.submission_metrics: Dict[str, Metric] = {}

        self.excluded_files = []
        self.required_files = []

        self.summary_metadata_csv_filepath = os.path.join(trojai_config.leaderboard_csvs_dirpath, '{}_METADATA.csv'.format(self.name))
        self.summary_results_csv_filepath = os.path.join(trojai_config.leaderboard_csvs_dirpath, '{}_RESULTS.csv'.format(self.name))

        self.dataset_manager = DatasetManager()
        self.task = None

    ########################
    # Utility IO functions
    ########################
    def save_json(self, trojai_config: TrojaiConfig):
        filepath = os.path.join(trojai_config.leaderboard_configs_dirpath, '{}_config.json'.format(self.name))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        json_io.write(filepath, self)

    @staticmethod
    def load_json(trojai_config: TrojaiConfig, name: str) -> Union['Leaderboard', None]:
        leaderboard_config_filepath = os.path.join(trojai_config.leaderboard_configs_dirpath, '{}_config.json'.format(name))

        if not os.path.exists(leaderboard_config_filepath):
            logging.error('Unable to find leaderboard config: {}'.format(leaderboard_config_filepath))
            return None

        leaderboard_config = json_io.read(leaderboard_config_filepath)
        assert leaderboard_config.task_name in Leaderboard.ALL_TASK_NAMES

        leaderboard_config.check_instance_data(trojai_config)

        return leaderboard_config

    def get_summary_schema_csv_filepath(self, trojai_config: TrojaiConfig):
        return os.path.join(trojai_config.summary_metrics_dirpath, '{}-schema-summary.csv'.format(self.name))

    def load_summary_results_csv_into_df(self):
        if os.path.exists(self.summary_results_csv_filepath):
            try:
                return pd.read_csv(self.summary_results_csv_filepath)
            except pd.errors.EmptyDataError:
                return None
        else:
            logging.warning('Unable to find summary results metadata_csv at location: {}, please generate it through the submission manager.'.format(self.summary_results_csv_filepath))
            return None

    def load_metadata_csv_into_df(self):
        if not os.path.exists(self.summary_metadata_csv_filepath):
            logging.warning('Unable to find summary metadata_csv at location: {}, generating CSV now.'.format(self.summary_metadata_csv_filepath))
            self.generate_metadata_csv()
        return pd.read_csv(self.summary_metadata_csv_filepath)

    def load_results_df(self, results_manager: ResultsManager):
        return results_manager.load_results(self.name, self.leaderboard_results_filepath, self.get_default_result_columns())

    def update_results_df(self, results_manager: ResultsManager, df):
        results_manager.update_results(self.name, df)

    ########################
    # Dataset utility functions
    ########################
    def get_default_prediction_result(self):
        return self.task.default_prediction_result
    def get_dataset(self, data_split_name):
        return self.dataset_manager.get(data_split_name)

    def has_dataset(self, data_split_name):
        return self.dataset_manager.has_dataset(data_split_name)

    def load_ground_truth(self, data_split_name) -> typing.OrderedDict[str, float]:
        raise NotImplementedError()

    def get_result_dirpath(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.results_dirpath

    def get_slurm_queue_name(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.slurm_queue_name

    def get_slurm_nice(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.slurm_nice

    def is_auto_delete_submission(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.auto_delete_submission

    def get_auto_execute_split_names(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.auto_execute_split_names

    def get_submission_window_time(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.submission_window_time_sec

    def can_submit_to_dataset(self, data_split_name: str):
        return self.dataset_manager.can_submit_to_dataset(data_split_name)

    def get_submission_data_split_names(self):
        return self.dataset_manager.get_submission_dataset_split_names()

    def get_html_data_split_names(self):
        return self.html_data_split_name_priorities.keys()

    def get_all_data_split_names(self):
        return self.dataset_manager.datasets.keys()

    def get_timeout_time_sec(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.timeout_time_sec

    def get_num_models(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.get_num_models()

    def check_instance_data(self, trojai_config: TrojaiConfig):
        has_updated = False
        has_updated = has_updated or self.task.check_instance_params(trojai_config)

        if not hasattr(self, 'check_for_missing_metrics'):
            has_updated = True
            self.check_for_missing_metrics = True

        if not hasattr(self, 'revision'):
            has_updated = True
            self.revision = 1

        if has_updated:
            self.save_json(trojai_config)

    def add_dataset(self, trojai_config: TrojaiConfig,
                    split_name: str,
                    can_submit: bool,
                    slurm_queue_name: str,
                    slurm_nice: int,
                    has_source_data: bool,
                    auto_delete_submission: bool,
                    auto_execute_split_names: list,
                    generate_metadata_csv: bool=False,
                    on_html: bool=False):
        if self.dataset_manager.has_dataset(split_name):
            raise RuntimeError('Dataset already exists in DatasetManager: {}'.format(split_name))

        dataset = Dataset(trojai_config, self.name, split_name, can_submit, slurm_queue_name, slurm_nice, has_source_data,
                           auto_delete_submission=auto_delete_submission, auto_execute_split_names=auto_execute_split_names)
        if self.task.verify_dataset(self.name, dataset, self.required_files):
            self.dataset_manager.add_dataset(dataset)
            if generate_metadata_csv:
                self.generate_metadata_csv(overwrite_csv=True)

            self.initialize_html_options(split_name, on_html)
            return True
        return False

    def initialize_directories(self):
        os.makedirs(self.submission_dirpath, exist_ok=True)
        self.dataset_manager.initialize_directories()

    def get_task(self) -> Task:
        return self.task

    def add_metric(self, metric: Metric):
        self.submission_metrics[metric.get_name()] = metric

    def remove_metric(self, metric_name: str):
        if metric_name not in self.submission_metrics:
            logging.info('Failed to remove {}, it does not exist'.format(metric_name))
        else:
            del self.submission_metrics[metric_name]

    def __str__(self):
        msg = 'Leaderboard: (\n'
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

    def write_html_leaderboard(self, is_trojai_accepting_submissions: bool, html_output_dirpath: str, is_first: bool, is_archived: bool):

        leaderboard_filename = '{}-leaderboard.html'.format(self.name)
        leaderboard_dirpath = os.path.join(html_output_dirpath, self.name)
        leaderboard_filepath = os.path.join(leaderboard_dirpath, leaderboard_filename)

        if not os.path.exists(leaderboard_dirpath):
            os.makedirs(leaderboard_dirpath, exist_ok=True)

        active_show = ''
        if is_first:
            active_show = 'active show'

        html_data_split_names = sorted(self.html_data_split_name_priorities, key=self.html_data_split_name_priorities.get, reverse=True)


        a = Airium()
        with a.div(klass='tab-pane fade {}'.format(active_show), id='{}'.format(self.name), role='tabpanel', **{'aria-labelledby' : 'tab-{}'.format(self.name)}):
            a('{{% include {}/about-{}.html %}}'.format(self.name, self.name))

            with a.div(klass='card-body card-body-cascade'):
                with a.p(klass='card-text text-left'):
                    for data_split in html_data_split_names:
                        if data_split in Leaderboard.DATASET_DESCRIPTIONS:
                            a('{}<br>'.format(Leaderboard.DATASET_DESCRIPTIONS[data_split]))

            with a.ul(klass='nav nav-pills', id='{}-tabs'.format(self.name), role='tablist'):
                with a.li(klass='nav-item'):
                    for data_split in html_data_split_names:
                        # TODO: For mitigation we might have to customize this to specify which data split to active show
                        if data_split == 'test':
                            a.a(klass='nav-link waves-light active show', id='tab-{}-{}'.format(self.name, data_split), href='#{}-{}'.format(self.name, data_split), **{'data-toggle': 'tab', 'aria-controls': '{}-{}'.format(self.name, data_split), 'aria-selected': 'true'}, _t=data_split)
                        else:
                            a.a(klass='nav-link waves-light', id='tab-{}-{}'.format(self.name, data_split), href='#{}-{}'.format(self.name, data_split), **{'data-toggle': 'tab', 'aria-controls': '{}-{}'.format(self.name, data_split), 'aria-selected': 'false'}, _t=data_split)

            with a.div(klass='tab-content card'):
                for data_split in html_data_split_names:
                    if not self.has_dataset(data_split):
                        continue

                    # TODO: For mitigation we might have to customize this to specify which data split to active show
                    if data_split == 'test':
                        active_show = 'active show'
                    else:
                        active_show = ''
                    with a.div(klass='tab-pane fade {}'.format(active_show), id='{}-{}'.format(self.name, data_split), role='tabpanel', **{'aria-labelledby': 'tab-{}-{}'.format(self.name, data_split)}):
                        with a.div(klass='card-body card-body-cascade'):
                            dataset = self.get_dataset(data_split)

                            required_format = 'Required filename format: "{}_{}_&lt;Submission Name&gt;.simg"'.format(self.name, data_split)
                            accepting_submissions_info = 'Accepting submissions: {}'.format(dataset.can_submit and not is_archived and is_trojai_accepting_submissions)
                            model_info = 'Number of models in {}, {}: {}'.format(self.name, data_split, self.get_num_models(data_split))
                            time_info = 'Execution timeout (hh:mm:ss): {}'.format(str(datetime.timedelta(seconds=self.get_timeout_time_sec(data_split))))

                            if is_archived:
                                a.p(klass='card-text text-left', _t='{}<br>{}'.format(accepting_submissions_info, model_info))
                            else:
                                a.p(klass='card-text text-left', _t='{}<br>{}<br>{}<br>{}'.format(required_format, accepting_submissions_info, model_info, time_info))

                        if not is_archived:
                            a('{{% include {}/jobs-{}-{}.html %}}'.format(self.name, self.name, data_split))

                        a('{{% include {}/results-unique-{}-{}.html %}}'.format(self.name, self.name, data_split))
                        a('{{% include {}/results-{}-{}.html %}}'.format(self.name, self.name, data_split))

        with open(leaderboard_filepath, 'w') as f:
            f.write(str(a))

        return leaderboard_filepath

    def initialize_html_options(self, split_name: str, on_html: bool):
        if on_html:
            self.html_data_split_name_priorities[split_name] = 0
        for table_name in Leaderboard.TABLE_NAMES:
            key = '{}-{}-{}'.format(self.name, split_name, table_name)
            if table_name == 'jobs':
                self.html_table_sort_options[key] = {'column': 'Submission Timestamp', 'order': 'desc',
                                                     'split_name': split_name}
            else:
                if split_name == 'sts':
                    self.html_table_sort_options[key] = {'column': 'Submission Timestamp', 'order': 'desc',
                                                         'split_name': split_name}
                else:
                    if self.dataset_manager.has_dataset(split_name) and self.evaluation_metric_name is not None:
                        sort_order = 'desc'
                        if self.evaluation_metric_name in self.submission_metrics:
                            sort_order = self.submission_metrics[self.evaluation_metric_name].get_sort_order()
                        self.html_table_sort_options[key] = {'column': self.evaluation_metric_name,
                                                             'order': sort_order, 'split_name': split_name}
                    else:
                        self.html_table_sort_options[key] = {'column': 'Submission Timestamp', 'order': 'asc',
                                                             'split_name': split_name}

    def generate_metadata_csv(self, overwrite_csv: bool = True):
        all_df_list = []

        if os.path.exists(self.summary_metadata_csv_filepath) and not overwrite_csv:
            logging.warning('Skipping building round metadata: {} already exists and overwrite is disabled.'.format(self.summary_metadata_csv_filepath))
            return

        for split_name in self.get_all_data_split_names():
            dataset = self.get_dataset(split_name)
            dataset_dirpath = dataset.dataset_dirpath

            metadata_filepath = os.path.join(dataset_dirpath, Dataset.METADATA_NAME)

            if not os.path.exists(metadata_filepath):
                logging.warning('Skipping {}, it does not contain the metadata file: {}'.format(dir, metadata_filepath))
                continue

            df = pd.read_csv(metadata_filepath)

            # Add column for data_split
            new_df = df.assign(data_split=split_name)

            # Add column for ground_truth
            new_df = new_df.assign(ground_truth='NaN')

            models_dir = os.path.join(dataset_dirpath, 'models')

            if os.path.exists(models_dir):
                # Add ground truth values into data
                for model_name in os.listdir(models_dir):
                    model_dirpath = os.path.join(models_dir, model_name)

                    # Skip model_name that is not a directory
                    if not os.path.isdir(model_dirpath):
                        continue

                    ground_truth_filepath = os.path.join(models_dir, model_name, Dataset.GROUND_TRUTH_NAME)
                    if not os.path.exists(ground_truth_filepath):
                        logging.warning('WARNING, ground truth file does not exist: {}'.format(ground_truth_filepath))
                        continue

                    with open(ground_truth_filepath, 'r') as f:
                        data = float(f.read())
                        new_df.loc[new_df['model_name'] == model_name, 'ground_truth'] = data
            else:
                logging.warning('{} does not exist'.format(models_dir))

            all_df_list.append(new_df)

        if len(all_df_list) == 0:
            logging.warning('Skipping {}, it does not contain any datasets for metadata'.format(self.summary_metadata_csv_filepath))
            return

        all_df = pd.concat(all_df_list)

        # Rearrange columns slightly
        columns = all_df.columns.tolist()
        column_order = ['model_name', 'ground_truth', 'data_split']
        remove_columns = ['converged', 'nonconverged_reason', 'Unnamed: 0']

        # Remove columns
        for column_name in remove_columns:
            if column_name in columns:
                columns.remove(column_name)

        # Reorder columns
        index = 0
        for column_name in column_order:
            if column_name in columns:
                columns.remove(column_name)
                columns.insert(index, column_name)
                index += 1

        all_df = all_df[columns]

        all_df.to_csv(self.summary_metadata_csv_filepath, index=False)
        logging.info('Finished writing round metadata to {}'.format(self.summary_metadata_csv_filepath))

    def get_default_result_columns(self):
        return ['submission_timestamp', 'actor_name', 'actor_UUID', 'data_split']

    def update_results_csv(self, result_df: pd.DataFrame, results_manager: ResultsManager, submission_epoch: int, data_split: str, actor_name: str, actor_uuid: str):
        raise NotImplementedError()

    def get_valid_metric(self, metric_name):
        raise NotImplementedError()

    def get_valid_summary_metric(self, metric_name):
        raise NotImplementedError()

    def get_training_dataset_name(self):
        raise NotImplementedError()

    def load_result(self, result_filepath):
        if os.path.exists(result_filepath):
            try:
                with open(result_filepath) as file:
                    file_contents = file.readline().strip()
                    return float(file_contents)
            except:
                return np.nan

        # If the file did not exist, then we return nan
        return np.nan

    def process_metrics(self, g_drive: DriveIO, results_manager: ResultsManager, data_split_name: str, execution_results_dirpath: str, actor_name: str, actor_uuid: str, submission_epoch_str: str, processed_metrics: list, skip_upload_existing: bool):
        raise NotImplementedError()

class TrojAILeaderboard(Leaderboard):
    DEFAULT_METRICS = [AverageCrossEntropy, CrossEntropyConfidenceInterval, BrierScore, ROC_AUC]
    DEFAULT_EVALUATION_METRIC_NAME = 'ROC-AUC'
    DEFAULT_EXCLUDED_FILES = ["detailed_stats.csv", "detailed_timing_stats.csv", "config.json", "ground_truth.csv", "log.txt", "log-per-class.txt", "machine.log", "poisoned-example-data", "stats.json", "METADATA.csv", "trigger_*", "DATA_LICENSE.txt", "METADATA_DICTIONARY.csv", "models-packaged", "README.txt", "watermark.json"]
    DEFAULT_REQUIRED_FILES = ["model.pt", "ground_truth.csv", "clean-example-data", "reduced-config.json"]
    TRAIN_DATASET_NAME = 'train'
    DEFAULT_DATASET_SPLIT_NAMES = ['train', 'test', 'sts', 'holdout', 'dev']
    DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES = ['train', 'test', 'sts', 'dev']
    VALID_TASK_NAMES = {'image_summary': ImageSummary,
                        'nlp_summary': NaturalLanguageProcessingSummary,
                        'image_classification': ImageClassification,
                        'image_object_detection' : ImageObjectDetection,
                        'image_segmentation': ImageSegmentation,
                        'nlp_sentiment': NaturalLanguageProcessingSentiment,
                        'nlp_ner': NaturalLanguageProcessingNamedEntityRecognition,
                        'nlp_qa': NaturalLanguageProcessingQuestionAnswering,
                        "cyber": CyberTask,
                        "cyber_apk_malware": CyberApkMalware,
                        "cyber_pdf_malware": CyberPdfMalware,
                        "rl_lavaworld": ReinforcementLearningLavaWorld,
                        "rl_gym": ReinforcementLearningSafetyGym,
                        'causal_language': CausalLanguageModeling,
                        "cyber_windows_pe_malware": CyberWindowsPEMalware,
                        "rl_colorful": ReinforcementLearningColorfulMemory
                        }

    VALID_METRIC_NAMES = {
        'AverageCrossEntropy': AverageCrossEntropy,
        'GroupedCrossEntropyViolin': GroupedCrossEntropyViolin,
        'GroupedROC-AUC': Grouped_ROC_AUC,
        'CrossEntropyConfidenceInterval': CrossEntropyConfidenceInterval,
        'BrierScore': BrierScore,
        'ROC_AUC': ROC_AUC,
        'DEX_Factor_csv': DEX_Factor_csv,
    }

    VALID_SUMMARY_METRIC_NAMES = {
        'SummaryAverageCEOverTime': SummaryAverageCEOverTime
    }

    DEFAULT_SUMMARY_METRICS = [SummaryAverageCEOverTime]


    def __init__(self, name: str, task_name: str, trojai_config: TrojaiConfig, add_default_data_split: bool = False, default_metrics = None, required_files=None):
        super().__init__(name, task_name, trojai_config)

        if self.task_name not in TrojAILeaderboard.VALID_TASK_NAMES:
            raise RuntimeError('Invalid task name: {}'.format(self.task_name))

        self.task = TrojAILeaderboard.VALID_TASK_NAMES[self.task_name](trojai_config, self.name)
        self.evaluation_metric_name = ROC_AUC().get_name()

        self.excluded_files.extend(TrojAILeaderboard.DEFAULT_EXCLUDED_FILES)

        if required_files is not None:
            self.required_files.extend(required_files)
        else:
            self.required_files.extend(TrojAILeaderboard.DEFAULT_REQUIRED_FILES)


        for metric in TrojAILeaderboard.DEFAULT_METRICS:
            new_metric = metric()
            self.submission_metrics[new_metric.get_name()] = new_metric

        for metric in TrojAILeaderboard.DEFAULT_SUMMARY_METRICS:
            new_metric = metric()
            self.summary_metrics.append(new_metric)

        if add_default_data_split:
            for split_name in TrojAILeaderboard.DEFAULT_DATASET_SPLIT_NAMES:
                if split_name in TrojAILeaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES:
                    can_submit = True
                    on_html = True
                else:
                    can_submit = False
                    on_html = False

                auto_delete_submission = False
                auto_execute_split_names = []

                if split_name == 'sts':
                    slurm_queue_name = TrojAILeaderboard.STS_SLURM_QUEUE_NAME
                    slurm_nice = 0
                    auto_delete_submission = True
                else:
                    slurm_queue_name = TrojAILeaderboard.GENERAL_SLURM_QUEUE_NAME
                    slurm_nice = 0

                if split_name == 'test':
                    auto_execute_split_names.append('train')

                source_data_filepath = os.path.join(trojai_config.datasets_dirpath, self.name, 'source-data')
                has_source_data = False

                if os.path.exists(source_data_filepath):
                    has_source_data = True

                self.add_dataset(trojai_config, split_name, can_submit, slurm_queue_name, slurm_nice, has_source_data,
                                 auto_delete_submission, auto_execute_split_names, generate_metadata_csv=False, on_html=on_html)

        self.initialize_directories()
        self.generate_metadata_csv()

    def get_default_result_columns(self):
        column_names = super().get_default_result_columns()
        column_names.extend(['model_names', 'raw_predictions', 'ground_truth'])
        for metric_name, metric in self.submission_metrics.items():
            if metric.store_result:
                column_names.append(metric_name)
        return column_names


    def get_training_dataset_name(self):
        return TrojAILeaderboard.TRAIN_DATASET_NAME


    def load_ground_truth(self, data_split_name: str) -> typing.OrderedDict[str, float]:
        dataset: Dataset = self.dataset_manager.get(data_split_name)

        # Dictionary storing ground truth data -- key = model name, value = answer/ground truth
        ground_truth_dict = collections.OrderedDict()

        models_dirpath: str = os.path.join(dataset.dataset_dirpath, Dataset.MODEL_DIRNAME)

        if os.path.exists(models_dirpath):
            for model_dir in os.listdir(models_dirpath):

                if not model_dir.startswith('id-'):
                    continue

                model_dirpath = os.path.join(models_dirpath, model_dir)

                if not os.path.isdir(model_dirpath):
                    continue

                ground_truth_file = os.path.join(model_dirpath, Dataset.GROUND_TRUTH_NAME)

                if not os.path.exists(ground_truth_file):
                    continue

                with open(ground_truth_file) as truth_file:
                    file_contents = truth_file.readline().strip()
                    ground_truth = float(file_contents)
                    ground_truth_dict[str(model_dir)] = ground_truth

        if len(ground_truth_dict) == 0:
            raise RuntimeError(
                'ground_truth_dict length was zero. No ground truth found in "{}"'.format(models_dirpath))

        return ground_truth_dict

    def update_results_csv(self, result_df: pd.DataFrame, results_manager: ResultsManager, submission_epoch: int, data_split: str, actor_name: str, actor_uuid: str):
        new_data = dict()

        submission_epoch_str = time_utils.convert_epoch_to_iso(submission_epoch)

        df = self.load_results_df(results_manager)
        filtered_df = results_manager.filter_primary_key(df, submission_epoch_str, data_split, actor_uuid)

        result_df_already_exists = result_df is not None and not result_df[
            (result_df['team_name'] == actor_name) & (result_df['submission_timestamp'] == submission_epoch_str) & (
                        result_df['data_split'] == data_split)].empty

        if result_df_already_exists:
            return new_data

        if filtered_df is None or len(filtered_df) != 1:
            logging.warning('Failed to find {}, {}, {}, when generating round results CSV'.format(
                submission_epoch_str, data_split, actor_uuid))
            return new_data

        model_names = filtered_df['model_names'].values[0]
        prediction_list = filtered_df['raw_predictions'].values[0]
        target_list = filtered_df['ground_truth'].values[0]

        # np_preds = np.array(prediction_list)
        # targets = np.array(target_list)

        if not result_df_already_exists:
            new_data['model_name'] =[str(i) for i in model_names]
            new_data['team_name'] = [actor_name] * len(model_names)
            new_data['submission_timestamp'] = [submission_epoch_str] * len(model_names)
            new_data['data_split'] = [data_split] * len(model_names)
            new_data['prediction'] = [float(i) for i in prediction_list]
            new_data['ground_truth'] = [float(i) for i in target_list]

        return new_data


    def process_metrics(self, g_drive: DriveIO, results_manager: ResultsManager, data_split_name: str,
                        execution_results_dirpath: str, actor_name: str, actor_uuid: str, submission_epoch_str: str, processed_metrics: list, skip_upload_existing: bool):
        # Initialize error strings to return
        errors = {}
        new_processed_metric_names = []
        web_display_parse_errors = ''

        # Load results dataframe
        df = self.load_results_df(results_manager)

        # Check to make sure that all metrics exist in the dataframe
        missing_columns = []
        for metric_name, metric in self.submission_metrics.items():
            if metric.store_result and metric_name not in df.columns:
                missing_columns.append(metric_name)

        # Add all missing metric columns, each with default value None
        if len(missing_columns) > 0:
            df = df.assign(**{col: None for col in missing_columns})

        #['submission_timestamp', 'actor_name', 'actor_UUID', 'data_split']
        filtered_df = results_manager.filter_primary_key(df, submission_epoch_str, data_split_name, actor_uuid)
        update_entry = {}
        metrics_to_compute = []

        # If the entry already exists, then we need to check for missing/empty metrics
        if filtered_df is not None:
            if len(filtered_df) > 1:
                logging.error('Found {} entries for submission {}, split {}, actor {}'.format(len(filtered_df), submission_epoch_str, data_split_name, actor_name))
                return

            submission_metric_names = self.submission_metrics.keys()

            # Check for metrics to compute
            for metric_name in submission_metric_names:
                if processed_metrics is None or metric_name not in processed_metrics:
                    metrics_to_compute.append(metric_name)
                # metric = self.submission_metrics[metric_name]
                # if metric.store_result:
                #     metric_values = filtered_df[metric_name].values[0]
                #     if metric_values is None:
                #         metrics_to_compute.append(metric_name)
                # elif (metric.share_with_actor or metric.share_with_external) and metric_name not in processed_metrics:
                #     metrics_to_compute.append(metric_name)

        # Entry is new, so we are creating a new row
        else:
            update_entry = {'submission_timestamp': submission_epoch_str, 'data_split': data_split_name,
                         'actor_UUID': actor_uuid, 'actor_name': actor_name}

            # Add all metrics to compute
            metrics_to_compute.extend(self.submission_metrics.keys())

        # If there are any metrics to compute, then let's compute them and update the filtered_df
        # We will also ensure we get the latest
        if len(metrics_to_compute) > 0:
            # Load ground truth, models names, and results from disk
            ground_truth = self.load_ground_truth(data_split_name)
            results = collections.OrderedDict()

            for model_name in ground_truth.keys():
                result_filepath = os.path.join(execution_results_dirpath, model_name + '.txt')
                result = self.load_result(result_filepath)

                if np.isnan(result) and ':Result Parse:' not in web_display_parse_errors:
                    web_display_parse_errors += ':Result Parse:'

                results[model_name] = result

            model_names = list(ground_truth.keys())
            model_names.sort()

            predictions = np.zeros(len(model_names), dtype=np.float64)
            targets = np.zeros(len(model_names), dtype=np.float64)

            for i in range(len(model_names)):
                predictions[i] = results[model_names[i]]
                targets[i] = ground_truth[model_names[i]]

            raw_predictions = copy.deepcopy(predictions)
            # ground_truth_list = targets.tolist()

            update_entry['model_names'] = model_names
            update_entry['raw_predictions'] = raw_predictions
            update_entry['ground_truth'] = targets

            # Check for issues with predictions and report them
            if not np.any(np.isfinite(predictions)):
                web_display_parse_errors += ":No Results:"

            num_missing_predictions = np.count_nonzero(np.isnan(predictions))

            if num_missing_predictions > 0:
                web_display_parse_errors += ":Missing Results:"

            # Update nan predictions with default value
            predictions[np.isnan(predictions)] = self.get_default_prediction_result()

            metadata_df = self.load_metadata_csv_into_df()
            data_split_metadata = metadata_df[metadata_df['data_split'] == data_split_name]

            external_share_files = []
            actor_share_files = []

            # Compute metrics (only TrojAIMetrics are compatible with TrojAILeaderboard)
            for metric_name in metrics_to_compute:
                metric = self.submission_metrics[metric_name]
                if isinstance(metric, TrojAIMetric):
                    metric_output = metric.compute(predictions, targets, model_names, data_split_metadata, actor_name, self.name, data_split_name, submission_epoch_str, execution_results_dirpath, skip_upload_existing)

                    new_processed_metric_names.append(metric_name)

                    if metric.store_result:
                        metric_result = metric_output['result']

                        if metric_result is not None:
                            update_entry[metric_name] = metric_result
                        else:
                            logging.warning('Metric {} is slated to return a result, but the result was None'.format(metric_name))

                    files = metric_output['files']

                    if files is not None and len(files) > 0:
                        if isinstance(files, str):
                            files = [files]

                        if metric.share_with_actor:
                            actor_share_files.extend(files)

                        if metric.share_with_external:
                            external_share_files.extend(files)
                else:
                    logging.warning('Invalid metric type: {}, expected TrojAIMetric for leaderboard {}'.format(type(metric), self.name))
                    continue

            # Update entry or add entry to result dataframe
            if filtered_df is not None:
                df.loc[filtered_df.index[0], update_entry.keys()] = update_entry.values()
            else:
                df.loc[len(df)] = update_entry

            # Upload metric files with external and actor Google Drive folders
            if len(external_share_files) > 0 or len(actor_share_files) > 0:
                actor_submission_folder_id, external_actor_submission_folder_id = g_drive.get_submission_actor_and_external_folder_ids(actor_name, self.name, data_split_name)

                if external_actor_submission_folder_id is not None:
                    # if len(external_share_files) > 0:
                    #     g_drive.enqueue_file_upload(external_share_files, folder_id=external_actor_submission_folder_id)
                    for file in external_share_files:
                        g_drive.upload(file, folder_id=external_actor_submission_folder_id, skip_existing=skip_upload_existing)

                if actor_submission_folder_id is not None:
                    # if len(actor_share_files) > 0:
                    #     g_drive.enqueue_file_upload(actor_share_files, folder_id=actor_submission_folder_id)
                    for file in actor_share_files:
                        g_drive.upload(file, folder_id=actor_submission_folder_id, skip_existing=skip_upload_existing)

        if len(web_display_parse_errors) != 0:
            errors['web_display_parse_errors'] = web_display_parse_errors

        self.update_results_df(results_manager, df)

        return errors, new_processed_metric_names

    def get_valid_metric(self, metric_name):
        return TrojAILeaderboard.VALID_METRIC_NAMES[metric_name]

    def get_valid_summary_metric(self, metric_name):
        return TrojAILeaderboard.VALID_SUMMARY_METRIC_NAMES[metric_name]


class MitigationLeaderboard(Leaderboard):
    DEFAULT_EVALUATION_METRIC_NAME = ''
    DEFAULT_EXCLUDED_FILES = ["test_example_data_lookup.json", 'ground_truth.csv', 'poisoned-example-data', 'detailed_stats.csv', 'fg_class_translation.json', 'clean-example-data', 'foregrounds',
                               "detailed_stats.csv", "detailed_timing_stats.csv", "config.json", "log.txt",
                               "log-per-class.txt", "machine.log", "poisoned-example-data", "stats.json", "METADATA.csv",
                               "trigger_*", "DATA_LICENSE.txt", "METADATA_DICTIONARY.csv", "models-packaged", "README.txt", "watermark.json"]
    DEFAULT_REQUIRED_FILES = ["model.pt", 'mitigate-example-data', "test-example-data", "reduced-config.json", "test_example_data_lookup.json", 'ground_truth.csv']
    TRAIN_DATASET_NAME = 'train'
    DEFAULT_DATASET_SPLIT_NAMES = ['train', 'test', 'sts', 'holdout', 'dev']
    DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES = ['train', 'test', 'sts', 'dev']
    VALID_TASK_NAMES = {'image_classification_mitigation': ImageClassificationMitigationTask}
    VALID_METRIC_NAMES = {
        'Mit_Psn_Acc': MitigationPoisonedAccuracyOnPoisonedModel,
        'Mit_Clean_Acc_Psn': MitigationCleanAccuracyOnPoisonedModel,
        'Mit_Clean_acc': MitigationAccuracyOnCleanModel,
        'Mit_Overall_acc': MitigationAccuraccyOverall,
        'Mit_fidelity': MitigationFidelityMetric
    }

    DEFAULT_METRICS = [MitigationPoisonedAccuracyOnPoisonedModel,
                       MitigationCleanAccuracyOnPoisonedModel,
                       MitigationAccuracyOnCleanModel]


    VALID_SUMMARY_METRIC_NAMES = {}
    DEFAULT_SUMMARY_METRICS = []
    def __init__(self, name: str, task_name: str, trojai_config: TrojaiConfig, add_default_data_split: bool, required_files=None):
        super().__init__(name, task_name, trojai_config)

        if self.task_name not in MitigationLeaderboard.VALID_TASK_NAMES:
            raise RuntimeError('Invalid task name: {}'.format(self.task_name))

        self.task = MitigationLeaderboard.VALID_TASK_NAMES[self.task_name](trojai_config, self.name)
        self.evaluation_metric_name = MitigationPoisonedAccuracyOnPoisonedModel().get_name()

        self.excluded_files.extend(MitigationLeaderboard.DEFAULT_EXCLUDED_FILES)

        if required_files is not None:
            self.required_files.extend(required_files)
        else:
            self.required_files.extend(MitigationLeaderboard.DEFAULT_REQUIRED_FILES)


        for metric in MitigationLeaderboard.DEFAULT_METRICS:
            new_metric = metric()
            self.submission_metrics[new_metric.get_name()] = new_metric

        for metric in MitigationLeaderboard.DEFAULT_SUMMARY_METRICS:
            new_metric = metric()
            self.summary_metrics.append(new_metric)

        if add_default_data_split:
            for split_name in MitigationLeaderboard.DEFAULT_DATASET_SPLIT_NAMES:
                if split_name in MitigationLeaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES:
                    can_submit = True
                    on_html = True
                else:
                    can_submit = False
                    on_html = False

                auto_delete_submission = False
                auto_execute_split_names = []

                if split_name == 'sts':
                    slurm_queue_name = MitigationLeaderboard.STS_SLURM_QUEUE_NAME
                    slurm_nice = 0
                    auto_delete_submission = True
                else:
                    slurm_queue_name = MitigationLeaderboard.GENERAL_SLURM_QUEUE_NAME
                    slurm_nice = 0

                has_source_data = False

                self.add_dataset(trojai_config, split_name, can_submit, slurm_queue_name, slurm_nice, has_source_data, auto_delete_submission, auto_execute_split_names, generate_metadata_csv=False, on_html=on_html)

        self.initialize_directories()
        self.generate_metadata_csv()

    def get_default_result_columns(self):
        column_names = super().get_default_result_columns()
        column_names.extend(['ground_truth_dict', 'predictions_dict'])
        for metric_name, metric in self.submission_metrics.items():
            if metric.store_result:
                column_names.append(metric_name)
        return column_names

    def get_training_dataset_name(self):
        return MitigationLeaderboard.TRAIN_DATASET_NAME

    # Loads the results from an actor into a dictionary with the format: Dict[model_name][example_filename] = np.nan or np.ndarray
    def load_results(self, actor_name: str, execution_results_dirpath: str, all_models_ground_truth: Dict[str, Dict[str, Any]], metadata_df: pd.DataFrame) -> Tuple[str, Dict[str, Dict[str, Union[float, np.ndarray]]]]:
        results_dict = {}
        errors = ''
        for model_name in all_models_ground_truth.keys():
            results_dict[model_name] = {}
            filtered_df = metadata_df[metadata_df['model_name'] == model_name]
            if len(filtered_df) != 1:
                logging.warning('Unable to find {} in metadata_df'.format(model_name))
                continue

            number_classes = filtered_df['number_classes'].values[0]

            example_file_list = all_models_ground_truth[model_name].keys()
            result_filepath = os.path.join(execution_results_dirpath, model_name + '.json')

            if not os.path.exists(result_filepath):
                for example_name in example_file_list:
                    results_dict[model_name][example_name] = np.nan
                logging.warning(
                    'Missing result file from actor {} for {} example {}'.format(actor_name,
                                                                                              model_name,
                                                                                              example_name))
                if ':Missing Results(file):' not in errors:
                    errors += ':Missing Results(file):'
                continue

            with open(result_filepath, 'r') as fp:
                prediction_dict = json.load(fp)

                for example_filename in example_file_list:
                    results_dict[model_name][example_filename] = np.nan
                    if example_filename in prediction_dict:
                        logits: np.ndarray = prediction_dict[example_filename]
                        if len(logits) == number_classes:
                            results_dict[model_name][example_filename] = logits
                        else:
                            results_dict[model_name][example_filename] = np.nan
                            if ':Missing Results(logits):' not in errors:
                                errors += ':Missing Results(logits):'

                            logging.warning('Missing results from actor {} for {} example {}, expected {}, got {}'.format(actor_name, model_name, example_filename, number_classes, len(logits)))

        return errors, results_dict


    def update_results_csv(self, result_df: pd.DataFrame, results_manager: ResultsManager, submission_epoch: int, data_split: str, actor_name: str, actor_uuid: str):
        new_data = dict()

        submission_epoch_str = time_utils.convert_epoch_to_iso(submission_epoch)

        df = self.load_results_df(results_manager)
        filtered_df = results_manager.filter_primary_key(df, submission_epoch_str, data_split, actor_uuid)

        result_df_already_exists = result_df is not None and not result_df[
            (result_df['team_name'] == actor_name) & (result_df['submission_timestamp'] == submission_epoch_str) & (
                        result_df['data_split'] == data_split)].empty


        if result_df_already_exists:
            return new_data

        if filtered_df is None or len(filtered_df) != 1:
            logging.warning('Failed to find {}, {}, {}, when generating round results CSV'.format(
                submission_epoch_str, data_split, actor_uuid))
            return new_data

        ground_truth_dict = filtered_df['ground_truth_dict'].values[0]
        predictions_dict = filtered_df['predictions_dict'].values[0]

        ground_truth_clean_entries = []
        ground_truth_poisoned_entries = []
        model_names = []
        example_names = []
        predictions = []


        for model_dir, examples_dict in ground_truth_dict.items():
            if examples_dict is None:
                continue
            for example_name, labels_dict in examples_dict.items():
                model_names.append(model_dir)
                example_names.append(example_name)
                ground_truth_clean_entries.append(labels_dict['clean'])
                ground_truth_poisoned_entries.append(labels_dict['poisoned'])
                logits = predictions_dict[model_dir][example_name]

                if logits is None or np.any(~(np.isfinite(logits))):
                    predictions.append(np.nan)
                else:
                    predictions.append(np.argmax(logits))



        if not result_df_already_exists:
            new_data['model_name'] = model_names
            new_data['team_name'] = [actor_name] * len(model_names)
            new_data['submission_timestamp'] = [submission_epoch_str] * len(model_names)
            new_data['data_split'] = [data_split] * len(model_names)
            new_data['example_name'] = example_names
            new_data['prediction'] = predictions
            new_data['ground_truth_clean_entry'] = ground_truth_clean_entries
            new_data['ground_truth_poisoned_entry'] = ground_truth_poisoned_entries

        return new_data

    # Loads the ground truth for every model and every example, this returns Dict[model_name][example_name]['clean'/'poisoned']
    def load_ground_truth(self, data_split_name) -> Dict[str, Dict[str, Dict[str, int]]]:
        # Tuple[
        # Dict[str, List[str]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        dataset: Dataset = self.dataset_manager.get(data_split_name)

        all_models_ground_truth = {}
        # clean_models_ground_truth = {}
        # poisoned_models_ground_truth = {}
        # poisoned_models_trigger_ground_truth = {}
        # all_model_examples = {}

        models_dirpath: str = os.path.join(dataset.dataset_dirpath, Dataset.MODEL_DIRNAME)

        if os.path.exists(models_dirpath):
            for model_dir in os.listdir(models_dirpath):
                if not model_dir.startswith('id-'):
                    continue

                model_dirpath = os.path.join(models_dirpath, model_dir)

                if not os.path.isdir(model_dirpath):
                    logging.warning('Failed to find model {}'.format(model_dirpath))
                    continue

                # model_ground_truth_filepath = os.path.join(model_dirpath, 'ground_truth.csv')
                # if not os.path.exists(model_ground_truth_filepath):
                #     logging.warning('Failed to find model ground truth CSV'.format(model_ground_truth_filepath))
                #     continue
                #
                # model_ground_truth = np.nan
                # with open(model_ground_truth_filepath, 'r') as fp:
                #     file_contents = fp.readline().strip()
                #     model_ground_truth = float(file_contents)

                test_data_ground_truth_filepath = os.path.join(model_dirpath, 'test_example_data_lookup.json')

                if not os.path.exists(test_data_ground_truth_filepath):
                    continue

                all_models_ground_truth[model_dir] = {}
                # clean_models_ground_truth[str(model_dir)] = {}
                # poisoned_models_trigger_ground_truth[str(model_dir)] = {}
                # poisoned_models_ground_truth[str(model_dir)] = {}
                # all_model_examples[model_dir] = []

                with open(test_data_ground_truth_filepath, 'r') as fp:
                    example_data_lookup_dict = json.load(fp)

                    if isinstance(example_data_lookup_dict, dict):
                        for example_filename, example_dict in example_data_lookup_dict.items():
                            # all_model_examples[model_dir].append(example_filename)
                            clean_label = example_dict['clean_label']
                            poisoned_label = example_dict['poisoned_label']

                            all_models_ground_truth[model_dir][example_filename] = {'clean': clean_label, 'poisoned': poisoned_label}
                            # if model_ground_truth == 0:
                            #     clean_models_ground_truth[str(model_dir)][example_filename] =  clean_label
                            # else:
                            #     poisoned_models_ground_truth[str(model_dir)][example_filename] = clean_label
                            #     poisoned_models_trigger_ground_truth[str(model_dir)][example_filename] = poisoned_label

        return all_models_ground_truth

    def get_valid_metric(self, metric_name):
        return MitigationLeaderboard.VALID_METRIC_NAMES[metric_name]

    def get_valid_summary_metric(self, metric_name):
        return MitigationLeaderboard.VALID_SUMMARY_METRIC_NAMES[metric_name]

    def process_metrics(self, g_drive: DriveIO, results_manager: ResultsManager, data_split_name: str,
                        execution_results_dirpath: str, actor_name: str, actor_uuid: str, submission_epoch_str: str,
                        processed_metrics: list, skip_upload_existing: bool):
        errors_dict = {}
        new_processed_metric_names = []
        web_display_parse_errors = ''

        # Load results dataframe
        df = self.load_results_df(results_manager)

        # Check to make sure that all metrics exist in the dataframe
        missing_columns = []
        for metric_name, metric in self.submission_metrics.items():
            if metric.store_result and metric_name not in df.columns:
                missing_columns.append(metric_name)

        # Add all missing metric columns, each with default value None
        if len(missing_columns) > 0:
            logging.info('Found missing columns that are being added to dataframe: {}'.format(missing_columns))
            df = df.assign(**{col: None for col in missing_columns})

        filtered_df = results_manager.filter_primary_key(df, submission_epoch_str, data_split_name, actor_uuid)

        update_entry = {}
        metrics_to_compute = []

        # If the entry already exists, then we need to check for missing/empty metrics
        if filtered_df is not None:
            if len(filtered_df) > 1:
                logging.error('Found {} entries for submission {}, split {}, actor {}'.format(len(filtered_df), submission_epoch_str, data_split_name, actor_name))
                return

            submission_metric_names = self.submission_metrics.keys()

            # Check for metrics to compute
            for metric_name in submission_metric_names:
                if processed_metrics is None or metric_name not in processed_metrics:
                    metrics_to_compute.append(metric_name)

        else:
            update_entry = {'submission_timestamp': submission_epoch_str, 'data_split': data_split_name,
                         'actor_UUID': actor_uuid, 'actor_name': actor_name}

            # Add all metrics to compute
            metrics_to_compute.extend(self.submission_metrics.keys())

        # If there are any metrics to compute, then let's compute them and update the filtered_df
        # We will also ensure we get the latest
        if len(metrics_to_compute) > 0:
            metadata_df = self.load_metadata_csv_into_df()
            data_split_metadata = metadata_df[metadata_df['data_split'] == data_split_name]

            all_models_ground_truth = self.load_ground_truth(data_split_name)
            errors, predictions_dict = self.load_results(actor_name, execution_results_dirpath, all_models_ground_truth, data_split_metadata)

            if errors != '':
                web_display_parse_errors = errors

            external_share_files = []
            actor_share_files = []

            update_entry['ground_truth_dict'] = all_models_ground_truth
            update_entry['predictions_dict'] = predictions_dict

            # Compute metrics for leaderboard
            for metric_name in metrics_to_compute:
                metric = self.submission_metrics[metric_name]

                if isinstance(metric, MitigationMetric):
                    metric_output = metric.compute(predictions_dict, all_models_ground_truth, data_split_metadata, actor_name, self.name, data_split_name, submission_epoch_str, execution_results_dirpath)

                    new_processed_metric_names.append(metric_name)

                    if metric.store_result:
                        metric_result = metric_output['result']

                        if metric_result is not None:
                            update_entry[metric_name] = metric_result
                        else:
                            logging.warning('{} Metric {} is slated to return a result, but the result was None'.format(self.name, metric_name))

                    files = metric_output['files']

                    if files is not None:
                        if isinstance(files, str):
                            files = [files]

                        if metric.share_with_actor:
                            actor_share_files.extend(files)

                        if metric.share_with_external:
                            external_share_files.extend(files)
                else:
                    logging.warning(
                        'Invalid metric type: {}, expected MitigationMetric for leaderboard {}'.format(type(metric),
                                                                                                   self.name))
                    continue

            # Update entry or add entry to result dataframe
            if filtered_df is not None:
                df.loc[filtered_df.index[0], update_entry.keys()] = update_entry.values()
            else:
                df.loc[len(df)] = update_entry

            # Upload metric files with external and actor Google Drive folders
            if len(external_share_files) > 0 or len(actor_share_files) > 0:
                actor_submission_folder_id, external_actor_submission_folder_id = g_drive.get_submission_actor_and_external_folder_ids(actor_name, self.name, data_split_name)

                if external_actor_submission_folder_id is not None:
                    # if len(external_share_files) > 0:
                    #     g_drive.enqueue_file_upload(external_share_files, folder_id=external_actor_submission_folder_id)
                    for file in external_share_files:
                        g_drive.upload(file, folder_id=external_actor_submission_folder_id, skip_existing=skip_upload_existing)

                if actor_submission_folder_id is not None:
                    # if len(actor_share_files) > 0:
                    #     g_drive.enqueue_file_upload(actor_share_files, folder_id=actor_submission_folder_id)
                    for file in actor_share_files:
                        g_drive.upload(file, folder_id=actor_submission_folder_id, skip_existing=skip_upload_existing)

        if len(web_display_parse_errors) != 0:
            errors_dict['web_display_parse_errors'] = web_display_parse_errors

        self.update_results_df(results_manager, df)

        return errors_dict, new_processed_metric_names


class LLMMitigationLeaderboard(Leaderboard):
    DEFAULT_EVALUATION_METRIC_NAME = ''
    DEFAULT_EXCLUDED_FILES = ["test_example_data_lookup.json", 'ground_truth.csv', 'poisoned-example-data', 'detailed_stats.csv', 'fg_class_translation.json', 'clean-example-data', 'foregrounds',
                               "detailed_stats.csv", "detailed_timing_stats.csv", "config.json", "log.txt",
                               "log-per-class.txt", "machine.log", "poisoned-example-data", "stats.json", "METADATA.csv",
                               "trigger_*", "DATA_LICENSE.txt", "METADATA_DICTIONARY.csv", "models-packaged", "README.txt", "watermark.json", 'detailed_test_eval_groundtruth.csv',
                              'eval-mmlu.json', 'generation_config.json', 'inference_log.txt', 'test_eval_groundtruth.json', 'test_eval_groundtruth_failed_responses.json', 'failed_responses.json',
                              'trojan-hparam.json', 'rot-hparams.json', 'all_results.json', 'eval_perplexity_stats.json', 'perplexity_log.txt', 'stats.json', 'training_args.json']
    DEFAULT_REQUIRED_FILES = ["model-00001-of-00004.safetensors", "test-example-data", "reduced-config.json", 'ground_truth.csv']
    TRAIN_DATASET_NAME = 'train'
    DEFAULT_DATASET_SPLIT_NAMES = ['train', 'test', 'sts', 'holdout', 'dev']
    DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES = ['train', 'test', 'sts', 'dev']
    VALID_TASK_NAMES = {'llm_mitigation': LLMMitigationTask}
    VALID_METRIC_NAMES = {
        'LLMMit_Avg_ASR': LLMMitigationAverageASR,
        'LLMMit_Avg_MMLU_PSN': LLMMitigationAveragePoisonedMMLU,
        'LLMMit_Avg_MMLU_Clean': LLMMitigationAverageCleanMMLU,
        'LLMMit_Avg_MMLU_All': LLMMitigationAverageAllMMLU,
        'LLMMit_Fidelity': LLMMitigationFidelityMetric

    }

    DEFAULT_METRICS = [LLMMitigationAverageASR,
                       LLMMitigationAveragePoisonedMMLU,
                       LLMMitigationAverageCleanMMLU,
                       LLMMitigationAverageAllMMLU,
                       LLMMitigationFidelityMetric]


    VALID_SUMMARY_METRIC_NAMES = {}
    DEFAULT_SUMMARY_METRICS = []
    def __init__(self, name: str, task_name: str, trojai_config: TrojaiConfig, add_default_data_split: bool, required_files=None):
        super().__init__(name, task_name, trojai_config)

        if self.task_name not in LLMMitigationLeaderboard.VALID_TASK_NAMES:
            raise RuntimeError('Invalid task name: {}'.format(self.task_name))

        self.task = LLMMitigationLeaderboard.VALID_TASK_NAMES[self.task_name](trojai_config, self.name)
        self.evaluation_metric_name = MitigationPoisonedAccuracyOnPoisonedModel().get_name()

        self.excluded_files.extend(LLMMitigationLeaderboard.DEFAULT_EXCLUDED_FILES)

        if required_files is not None:
            self.required_files.extend(required_files)
        else:
            self.required_files.extend(LLMMitigationLeaderboard.DEFAULT_REQUIRED_FILES)


        for metric in LLMMitigationLeaderboard.DEFAULT_METRICS:
            new_metric = metric()
            self.submission_metrics[new_metric.get_name()] = new_metric

        for metric in LLMMitigationLeaderboard.DEFAULT_SUMMARY_METRICS:
            new_metric = metric()
            self.summary_metrics.append(new_metric)

        if add_default_data_split:
            for split_name in LLMMitigationLeaderboard.DEFAULT_DATASET_SPLIT_NAMES:
                if split_name in LLMMitigationLeaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES:
                    can_submit = True
                    on_html = True
                else:
                    can_submit = False
                    on_html = False

                auto_delete_submission = False
                auto_execute_split_names = []

                if split_name == 'sts':
                    slurm_queue_name = LLMMitigationLeaderboard.STS_SLURM_QUEUE_NAME
                    slurm_nice = 0
                    auto_delete_submission = True
                else:
                    slurm_queue_name = LLMMitigationLeaderboard.GENERAL_SLURM_QUEUE_NAME
                    slurm_nice = 0

                source_data_filepath = os.path.join(trojai_config.datasets_dirpath, self.name, 'source-data')
                has_source_data = False

                if os.path.exists(source_data_filepath):
                    has_source_data = True

                self.add_dataset(trojai_config, split_name, can_submit, slurm_queue_name, slurm_nice, has_source_data, auto_delete_submission, auto_execute_split_names, generate_metadata_csv=False, on_html=on_html)

        self.initialize_directories()
        self.generate_metadata_csv()

    def get_default_result_columns(self):
        column_names = super().get_default_result_columns()

        column_names.extend(['ground_truth_asr', 'ground_truth_mmlu', 'mitigated_asr', 'mitigated_mmlu'])

        for metric_name, metric in self.submission_metrics.items():
            if metric.store_result:
                column_names.append(metric_name)
        return column_names

    def get_training_dataset_name(self):
        return LLMMitigationLeaderboard.TRAIN_DATASET_NAME

    # Loads the results from an actor into a dictionary with the format: Dict[model_name][example_filename] = np.nan or np.ndarray
    def load_results(self, actor_name: str, execution_results_dirpath: str, all_models_ground_truth: Dict[str, Dict[str, Any]], metadata_df: pd.DataFrame) -> Tuple[str, Dict[str, Dict[str, Union[float, np.ndarray]]]]:
        results_dict = {}
        errors = ''
        for model_name in all_models_ground_truth.keys():
            results_dict[model_name] = {}
            filtered_df = metadata_df[metadata_df['model_name'] == model_name]
            if len(filtered_df) != 1:
                logging.warning('Unable to find {} in metadata_df'.format(model_name))
                continue

            results_dict[model_name]['mmlu'] = np.nan
            results_dict[model_name]['asr'] = np.nan

            result_filepath = os.path.join(execution_results_dirpath, model_name + '.json')

            if not os.path.exists(result_filepath):
                logging.warning(
                    'Missing result file from actor {} for {}'.format(actor_name, model_name))
                if ':Missing Results(file):' not in errors:
                    errors += ':Missing Results(file):'
                continue


            with open(result_filepath, 'r') as fp:
                result_dict = json.load(fp)
                if 'mmlu' in result_dict:
                    results_dict[model_name]['mmlu'] = result_dict['mmlu']

                if 'asr' in result_dict:
                    results_dict[model_name]['asr'] = result_dict['asr']
        return errors, results_dict


    def update_results_csv(self, result_df: pd.DataFrame, results_manager: ResultsManager, submission_epoch: int, data_split: str, actor_name: str, actor_uuid: str):
        new_data = dict()

        submission_epoch_str = time_utils.convert_epoch_to_iso(submission_epoch)

        df = self.load_results_df(results_manager)
        filtered_df = results_manager.filter_primary_key(df, submission_epoch_str, data_split, actor_uuid)

        result_df_already_exists = result_df is not None and not result_df[
            (result_df['team_name'] == actor_name) & (result_df['submission_timestamp'] == submission_epoch_str) & (
                        result_df['data_split'] == data_split)].empty


        if result_df_already_exists:
            return new_data

        if filtered_df is None or len(filtered_df) != 1:
            logging.warning('Failed to find {}, {}, {}, when generating round results CSV'.format(
                submission_epoch_str, data_split, actor_uuid))
            return new_data

        model_names = filtered_df['model_names'].values[0]
        gt_asr_value = filtered_df['ground_truth_asr'].values[0]
        gt_mmlu_value = filtered_df['ground_truth_mmlu'].values[0]
        mit_asr_value = filtered_df['mitigated_asr'].values[0]
        mit_mmlu_value = filtered_df['mitigated_mmlu'].values[0]

        if not result_df_already_exists:
            new_data['model_name'] = model_names
            new_data['team_name'] = [actor_name] * len(model_names)
            new_data['submission_timestamp'] = [submission_epoch_str] * len(model_names)
            new_data['data_split'] = [data_split] * len(model_names)
            new_data['ground_truth_asr'] = gt_asr_value
            new_data['ground_truth_mmlu'] = gt_mmlu_value
            new_data['mitigated_truth_asr'] = mit_asr_value
            new_data['mitigated_truth_mmlu'] = mit_mmlu_value

        return new_data

    # Loads the ground truth for every model and every example, this returns Dict[model_name][example_name]['clean'/'poisoned']
    def load_ground_truth(self, data_split_name) -> Dict[str, Dict[str, float]]:
        # Tuple[
        # Dict[str, List[str]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        dataset: Dataset = self.dataset_manager.get(data_split_name)

        all_models_ground_truth = {}
        # clean_models_ground_truth = {}
        # poisoned_models_ground_truth = {}
        # poisoned_models_trigger_ground_truth = {}
        # all_model_examples = {}

        models_dirpath: str = os.path.join(dataset.dataset_dirpath, Dataset.MODEL_DIRNAME)

        if os.path.exists(models_dirpath):
            for model_dir in os.listdir(models_dirpath):
                if not model_dir.startswith('id-'):
                    continue

                model_dirpath = os.path.join(models_dirpath, model_dir)

                if not os.path.isdir(model_dirpath):
                    logging.warning('Failed to find model {}'.format(model_dirpath))
                    continue

                test_data_ground_truth_filepath = os.path.join(model_dirpath, 'test_eval_groundtruth.json')
                test_data_ground_truth_dict = None
                if os.path.exists(test_data_ground_truth_filepath):
                    with open(test_data_ground_truth_filepath, 'r') as fp:
                        test_data_ground_truth_dict = json.load(fp)

                mmlu_filepath = os.path.join(model_dirpath, 'eval-mmlu.json')
                if not os.path.exists(mmlu_filepath):
                    logging.warning('Failed to find model {} mmlu'.format(model_dirpath))
                    continue

                mmlu_value = None
                with open(mmlu_filepath, 'r') as fp:
                    mmlu_dict = json.load(fp)
                    mmlu_value = float(mmlu_dict['results']['mmlu']['acc,none'])

                asr_value = None
                if test_data_ground_truth_dict is not None and 'all_trigger0_asr' in test_data_ground_truth_dict:
                    asr_value = test_data_ground_truth_dict['all_trigger0_asr']

                all_models_ground_truth[model_dir] = {'asr': asr_value, 'mmlu': mmlu_value}

        return all_models_ground_truth

    def get_valid_metric(self, metric_name):
        return LLMMitigationLeaderboard.VALID_METRIC_NAMES[metric_name]

    def get_valid_summary_metric(self, metric_name):
        return LLMMitigationLeaderboard.VALID_SUMMARY_METRIC_NAMES[metric_name]

    def process_metrics(self, g_drive: DriveIO, results_manager: ResultsManager, data_split_name: str,
                        execution_results_dirpath: str, actor_name: str, actor_uuid: str, submission_epoch_str: str,
                        processed_metrics: list, skip_upload_existing: bool):
        errors_dict = {}
        new_processed_metric_names = []
        web_display_parse_errors = ''

        # Load results dataframe
        df = self.load_results_df(results_manager)

        # Check to make sure that all metrics exist in the dataframe
        missing_columns = []
        for metric_name, metric in self.submission_metrics.items():
            if metric.store_result and metric_name not in df.columns:
                missing_columns.append(metric_name)

        # Add all missing metric columns, each with default value None
        if len(missing_columns) > 0:
            logging.info('Found missing columns that are being added to dataframe: {}'.format(missing_columns))
            df = df.assign(**{col: None for col in missing_columns})

        filtered_df = results_manager.filter_primary_key(df, submission_epoch_str, data_split_name, actor_uuid)

        update_entry = {}
        metrics_to_compute = []

        # If the entry already exists, then we need to check for missing/empty metrics
        if filtered_df is not None:
            if len(filtered_df) > 1:
                logging.error('Found {} entries for submission {}, split {}, actor {}'.format(len(filtered_df), submission_epoch_str, data_split_name, actor_name))
                return

            submission_metric_names = self.submission_metrics.keys()

            # Check for metrics to compute
            for metric_name in submission_metric_names:
                if processed_metrics is None or metric_name not in processed_metrics:
                    metrics_to_compute.append(metric_name)

        else:
            update_entry = {'submission_timestamp': submission_epoch_str, 'data_split': data_split_name,
                         'actor_UUID': actor_uuid, 'actor_name': actor_name}

            # Add all metrics to compute
            metrics_to_compute.extend(self.submission_metrics.keys())

        # If there are any metrics to compute, then let's compute them and update the filtered_df
        # We will also ensure we get the latest
        if len(metrics_to_compute) > 0:
            metadata_df = self.load_metadata_csv_into_df()
            data_split_metadata = metadata_df[metadata_df['data_split'] == data_split_name]

            all_models_ground_truth = self.load_ground_truth(data_split_name)
            errors, predictions_dict = self.load_results(actor_name, execution_results_dirpath, all_models_ground_truth, data_split_metadata)

            if errors != '':
                web_display_parse_errors = errors

            external_share_files = []
            actor_share_files = []

            update_entry['ground_truth_asr'] = all_models_ground_truth['asr']
            update_entry['ground_truth_mmlu'] = all_models_ground_truth['mmlu']
            update_entry['mitigated_asr'] = predictions_dict['asr']
            update_entry['mitigated_mmlu'] = predictions_dict['mmlu']

            # Compute metrics for leaderboard
            for metric_name in metrics_to_compute:
                metric = self.submission_metrics[metric_name]

                if isinstance(metric, LLMMitigationMetric):
                    metric_output = metric.compute(predictions_dict, all_models_ground_truth, data_split_metadata, actor_name, self.name, data_split_name, submission_epoch_str, execution_results_dirpath)

                    new_processed_metric_names.append(metric_name)

                    if metric.store_result:
                        metric_result = metric_output['result']

                        if metric_result is not None:
                            update_entry[metric_name] = metric_result
                        else:
                            logging.warning('{} Metric {} is slated to return a result, but the result was None'.format(self.name, metric_name))

                    files = metric_output['files']

                    if files is not None:
                        if isinstance(files, str):
                            files = [files]

                        if metric.share_with_actor:
                            actor_share_files.extend(files)

                        if metric.share_with_external:
                            external_share_files.extend(files)
                else:
                    logging.warning(
                        'Invalid metric type: {}, expected LLMMitigationMetric for leaderboard {}'.format(type(metric),
                                                                                                   self.name))
                    continue

            # Update entry or add entry to result dataframe
            if filtered_df is not None:
                df.loc[filtered_df.index[0], update_entry.keys()] = update_entry.values()
            else:
                df.loc[len(df)] = update_entry

            # Upload metric files with external and actor Google Drive folders
            if len(external_share_files) > 0 or len(actor_share_files) > 0:
                actor_submission_folder_id, external_actor_submission_folder_id = g_drive.get_submission_actor_and_external_folder_ids(actor_name, self.name, data_split_name)

                if external_actor_submission_folder_id is not None:
                    # if len(external_share_files) > 0:
                    #     g_drive.enqueue_file_upload(external_share_files, folder_id=external_actor_submission_folder_id)
                    for file in external_share_files:
                        g_drive.upload(file, folder_id=external_actor_submission_folder_id, skip_existing=skip_upload_existing)

                if actor_submission_folder_id is not None:
                    # if len(actor_share_files) > 0:
                    #     g_drive.enqueue_file_upload(actor_share_files, folder_id=actor_submission_folder_id)
                    for file in actor_share_files:
                        g_drive.upload(file, folder_id=actor_submission_folder_id, skip_existing=skip_upload_existing)

        if len(web_display_parse_errors) != 0:
            errors_dict['web_display_parse_errors'] = web_display_parse_errors

        self.update_results_df(results_manager, df)

        return errors_dict, new_processed_metric_names


def init_leaderboard(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    required_files = None
    if args.required_files is not None:
        required_files = args.required_files.split(',')

    leaderboard_type = args.leaderboard_type

    leaderboard = None

    if leaderboard_type == 'trojai':
        leaderboard = TrojAILeaderboard(args.name, args.task_name, trojai_config, add_default_data_split=args.add_default_datasplit, required_files=required_files)
    elif leaderboard_type == 'mitigation':
        leaderboard = MitigationLeaderboard(args.name, args.task_name, trojai_config, add_default_data_split=args.add_default_datasplit, required_files=required_files)
    elif leaderboard_type == 'llm_mitigation':
        leaderboard = LLMMitigationLeaderboard(args.name, args.task_name, trojai_config, add_default_data_split=args.add_default_datasplit, required_files=required_files)

    if leaderboard is not None:
        leaderboard.save_json(trojai_config)
        print('Created: {}'.format(leaderboard))

def add_dataset_to_leaderboard(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    if args.slurm_queue_name is None:
        slurm_queue_name = Leaderboard.GENERAL_SLURM_QUEUE_NAME
    else:
        slurm_queue_name = args.slurm_queue_name

    leaderboard = Leaderboard.load_json(trojai_config, args.name)

    if args.split_name == 'sts':
        auto_delete_submission = True
    else:
        auto_delete_submission = False

    if leaderboard.add_dataset(trojai_config, args.split_name, args.can_submit, slurm_queue_name, args.slurm_nice, args.has_source_data, auto_delete_submission, [], on_html=args.on_html):
        leaderboard.generate_metadata_csv(overwrite_csv=True)
        leaderboard.save_json(trojai_config)

        print('Added dataset {} to {}'.format(args.split_name, args.name))
    else:
        print('Failed to add dataset')


def generate_summary_metadata(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    leaderboard = Leaderboard.load_json(trojai_config, args.name)
    leaderboard.generate_metadata_csv(overwrite_csv=True)


def add_metric(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    leaderboard = Leaderboard.load_json(trojai_config, args.name)
    metric_name = args.metric_name

    metric_params_filepath = args.metric_params_json_filepath
    metric_param_dict = None

    if metric_params_filepath is not None:
        try:
            with open(metric_params_filepath, mode='r', encoding='utf-8') as f:
                metric_param_dict = json.load(f)
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(metric_params_filepath))
            raise
        except:
            raise

    if metric_param_dict is None:
        new_metric = leaderboard.get_valid_metric(metric_name)()
    else:
        new_metric = leaderboard.get_valid_metric(metric_name)(**metric_param_dict)

    leaderboard.add_metric(new_metric)
    leaderboard.save_json(trojai_config)

    print('Added metric {} to {}'.format(metric_name, leaderboard.name))

def remove_metric(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    leaderboard = Leaderboard.load_json(trojai_config, args.name)
    metric_name = args.metric_name

    leaderboard.remove_metric(metric_name)
    leaderboard.save_json(trojai_config)

    print('Removed metric {} to {}'.format(metric_name, leaderboard.name))

def add_summary_metric(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    leaderboard = Leaderboard.load_json(trojai_config, args.name)
    metric_name = args.metric_name

    metric_params_filepath = args.metric_params_json_filepath
    metric_param_dict = None

    if metric_params_filepath is not None:
        try:
            with open(metric_params_filepath, mode='r', encoding='utf-8') as f:
                metric_param_dict = json.load(f)
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(metric_params_filepath))
            raise
        except:
            raise

    if metric_param_dict is None:
        new_metric = leaderboard.get_valid_summary_metric(metric_name)()
    else:
        new_metric = leaderboard.get_valid_summary_metric(metric_name)(**metric_param_dict)

    leaderboard.summary_metrics.append(new_metric)
    leaderboard.save_json(trojai_config)

    print('Added summary metric {} to {}'.format(metric_name, leaderboard.name))



def update_configuration_latest(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    leaderboard_names = []

    if args.name is not None:
        leaderboard_names.append(args.name)
    else:
        for name in trojai_config.archive_leaderboard_names:
            leaderboard_names.append(name)
        for name in trojai_config.active_leaderboard_names:
            leaderboard_names.append(name)

    for name in leaderboard_names:
        leaderboard_config_filepath = os.path.join(trojai_config.leaderboard_configs_dirpath, '{}_config.json'.format(name))

        backup_filepath = os.path.join(trojai_config.leaderboard_configs_dirpath, '{}_config_backup.json'.format(name))

        if not os.path.exists(leaderboard_config_filepath):
            print('Warning: Unable to find {}, skipping...'.format(leaderboard_config_filepath))
            continue

        # Load leaderboard config using json
        old_leaderboard_config = None
        with open(leaderboard_config_filepath, 'r') as fp:
            old_leaderboard_config = json.load(fp)

        if os.path.exists(backup_filepath):
            print('Error, backup already exists, skipping {}'.format(name))
            continue
            # return

        # Write backup
        with open(backup_filepath, 'w') as fp:
            json.dump(old_leaderboard_config, fp, indent=2)

        leaderboard_name = None
        task_name = None
        py_object = None

        if 'py/object' in old_leaderboard_config:
            py_object = old_leaderboard_config['py/object']

        if 'name' in old_leaderboard_config:
            leaderboard_name = old_leaderboard_config['name']

        if 'task_name' in old_leaderboard_config:
            task_name = old_leaderboard_config['task_name']

        if py_object is None or leaderboard_name is None or task_name is None:
            print('Error unable to parse old leaderboard config {}, skipping...'.format(leaderboard_config_filepath))
            continue

        new_leaderboard_inst = None

        if py_object == 'leaderboards.leaderboard.Leaderboard':
            new_leaderboard_inst = TrojAILeaderboard(leaderboard_name, task_name, trojai_config)
        elif py_object == '__main__.Leaderboard':
            new_leaderboard_inst = TrojAILeaderboard(leaderboard_name, task_name, trojai_config)
        # elif py_object == '__main__.TrojAILeaderboard':
        #     new_leaderboard_inst = TrojAILeaderboard(leaderboard_name, task_name, trojai_config)
        # elif py_object == 'leaderboards.leaderboard.TrojAILeaderboard':
        #     new_leaderboard_inst = TrojAILeaderboard(leaderboard_name, task_name, trojai_config)
        else:
            print('Error unknown py_object: {} for {}, skipping...'.format(py_object, leaderboard_config_filepath))
            continue

        # First let's work with the task object
        if 'task' in old_leaderboard_config:
            task_copy = copy.deepcopy(new_leaderboard_inst.task)
            old_task_dict = old_leaderboard_config['task']
            del old_task_dict['py/object']

            update_object_values(task_copy, old_task_dict)
            new_leaderboard_inst.task = task_copy

        # Next work with the dataset manager and corresponding datasets, metrics, excluded and required files
        updated_submission_metrics = False
        updated_excluded_files = False
        updated_required_files = False

        if 'dataset_manager' in old_leaderboard_config:
            old_dataset_manager = old_leaderboard_config['dataset_manager']

            for dataset_name in old_dataset_manager['datasets'].keys():
                old_dataset_dict = old_dataset_manager['datasets'][dataset_name]

                split_name = get_value(old_dataset_dict, 'split_name')
                can_submit = get_value(old_dataset_dict, 'can_submit')
                slurm_queue_name = get_value(old_dataset_dict, 'slurm_queue_name')
                slurm_nice = get_value(old_dataset_dict, 'slurm_nice')
                has_source_data = False
                if 'source_dataset_dirpath' in old_dataset_dict:
                    has_source_data = True

                if split_name is None or can_submit is None or slurm_queue_name is None or slurm_nice is None:
                    print('Error, unable to process dataset {} for {}'.format(dataset_name, leaderboard_config_filepath))
                    continue

                new_dataset = Dataset(trojai_config, new_leaderboard_inst.name, split_name, can_submit, slurm_queue_name, slurm_nice, has_source_data)

                update_object_values(new_dataset, old_dataset_dict)

                new_leaderboard_inst.dataset_manager.add_dataset(new_dataset)

                if not updated_submission_metrics:
                    if 'submission_metrics' in old_dataset_dict:
                        for metric_name, metric_dict in old_dataset_dict['submission_metrics'].items():
                            # Fix for renaming the store_result attribute in Metric class
                            if 'store_result_in_submission' in metric_dict:
                                temp_store_result = copy.deepcopy(metric_dict['store_result_in_submission'])
                                del metric_dict['store_result_in_submission']
                                metric_dict['store_result'] = temp_store_result
                            json_string = json.dumps(metric_dict)

                            metric_inst = jsonpickle.decode(json_string)


                            new_leaderboard_inst.submission_metrics[metric_name] = metric_inst
                        updated_submission_metrics = True

                if not updated_excluded_files:
                    if 'excluded_files' in old_dataset_dict:
                        new_leaderboard_inst.excluded_files = copy.deepcopy(old_dataset_dict['excluded_files'])
                        updated_excluded_files = True

                if not updated_required_files:
                    if 'required_files' in old_dataset_dict:
                        new_leaderboard_inst.required_files = copy.deepcopy(old_dataset_dict['required_files'])
                        updated_required_files = True

        # handle summary metrics
        if 'summary_metrics' in old_leaderboard_config:
            old_summary_metrics = old_leaderboard_config['summary_metrics']
            for old_summary_metric_dict in old_summary_metrics:
                json_string = json.dumps(old_summary_metric_dict)
                metric_inst = jsonpickle.decode(json_string)
                new_leaderboard_inst.summary_metrics.append(metric_inst)


        # hold out task and dataset objects and metrics
        del old_leaderboard_config['task']
        del old_leaderboard_config['dataset_manager']
        del old_leaderboard_config['summary_metrics']

        update_object_values(new_leaderboard_inst, old_leaderboard_config)

        new_leaderboard_inst.save_json(trojai_config)
        print('Finished updating leaderboard {}'.format(name))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Runs leaderboards commands')
    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    init_parser = subparser.add_parser('init', help='Initializes a new leaderboard')
    init_parser.add_argument('--leaderboard-type', type=str, choices=Leaderboard.LEADERBOARD_TYPES, help='Declares the type of leaderboard to generate')
    init_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    init_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    init_parser.add_argument('--task-name', type=str, choices=Leaderboard.ALL_TASK_NAMES, help='The name of the task for this leaderboards', required=True)
    init_parser.add_argument('--required-files', type=str, default=None, help='The set of required files, defaults to the defaults set if not used. Tis is a csv list like " --required-files=model.pt,test.sh,img.png"')
    init_parser.add_argument('--add-default-datasplit', help='Will attempt to add the default data splits, if they fail task checks then will not be added. Need to call add-dataset when they are ready.', action='store_true')
    init_parser.set_defaults(func=init_leaderboard)

    add_dataset_parser = subparser.add_parser('add-dataset', help='Adds a dataset into a leaderboard')
    add_dataset_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    add_dataset_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    add_dataset_parser.add_argument('--split-name', type=str, help='The dataset split name', required=True)
    add_dataset_parser.add_argument('--has-source-data', action='store_true', help='Indicates that the dataset has source data that is saved on disk, format: "leaderboard_name-source_data"')
    add_dataset_parser.add_argument('--can-submit', action='store_true', help='Whether actors can submit to the dataset')
    add_dataset_parser.add_argument('--slurm-queue-name', type=str, help='The name of the slurm queue')
    add_dataset_parser.add_argument('--slurm-nice', type=int, help='The nice value when launching jobs for this dataset (0 is highest priority)', default=0)
    add_dataset_parser.add_argument('--on-html', action='store_true', help='Whether the dataset will be shown on the HTML page')
    add_dataset_parser.set_defaults(func=add_dataset_to_leaderboard)

    summary_results_parser = subparser.add_parser('generate_summary_metadata', help='Generates the METADATA CSV file for the leaderboard')
    summary_results_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config',required=True)
    summary_results_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    summary_results_parser.set_defaults(func=generate_summary_metadata)

    add_metric_parser = subparser.add_parser('add-metric', help='Adds metric to leaderboard')
    add_metric_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    add_metric_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    add_metric_parser.add_argument('--metric-name', type=str, choices=Leaderboard.ALL_METRIC_NAMES, help='The name of the metric to add', required=True)
    add_metric_parser.add_argument('--metric-params-json-filepath', type=str, help='The filepath to the json file that describes custom metric parameters', default=None)
    add_metric_parser.set_defaults(func=add_metric)

    remove_metric_parser = subparser.add_parser('remove-metric', help='Removes metric from leaderboard')
    remove_metric_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    remove_metric_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    remove_metric_parser.add_argument('--metric-name', type=str, help='The name of the metric to remove', required=True)
    remove_metric_parser.set_defaults(func=remove_metric)

    add_summary_metric_parser = subparser.add_parser('add-summary-metric', help='Adds a summary metric to leaderboard')
    add_summary_metric_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    add_summary_metric_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    add_summary_metric_parser.add_argument('--metric-name', type=str, choices=Leaderboard.ALL_SUMMARY_METRIC_NAMES, help='The name of the metric to add', required=True)
    add_summary_metric_parser.add_argument('--metric-params-json-filepath', type=str, help='The filepath to the json file that describes custom metric parameters', default=None)
    add_summary_metric_parser.set_defaults(func=add_summary_metric)

    update_config_parser = subparser.add_parser('update-config')
    update_config_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    update_config_parser.add_argument('--name', type=str, help='The name of the leaderboard to update, if not specified then will update all leaderboards', required=False, default=None)
    update_config_parser.set_defaults(func=update_configuration_latest)

    args = parser.parse_args()
    args.func(args)

