# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.


# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import copy
import itertools
import json

import numpy as np
from sklearn.metrics import auc
import pandas as pd

import matplotlib.pyplot as plt

import os

from leaderboards import fs_utils
import metadata_utils



class Metric(object):

    def __init__(self, write_html: bool, share_with_actor: bool, store_result_in_submission: bool, share_with_external: bool):
        self.write_html = write_html
        self.share_with_actor = share_with_actor
        self.store_result_in_submission = store_result_in_submission
        self.share_with_external = share_with_external
        self.html_priority = 0
        self.html_decimal_places = 5

    def get_name(self):
        raise NotImplementedError()

    # Returns a dictionary with the following:
    # 'result': None or value
    # 'metadata': None or dict
    # 'files': None or list of files saved
    def compute(self, predictions: np.ndarray, targets: np.ndarray, model_names: list, metadata_df: pd.DataFrame, actor_name: str, leaderboard_name: str, data_split_name: str, submission_epoch_str: str, output_dirpath: str):
        raise NotImplementedError()

    def compare(self, computed, baseline):
        raise NotImplementedError()

class AverageCrossEntropy(Metric):
    def __init__(self, write_html:bool = True, share_with_actor:bool = False, store_result_in_submission:bool = True, share_with_external: bool = False, epsilon:float = 1e-12):
        super().__init__(write_html, share_with_actor, store_result_in_submission, share_with_external)
        self.epsilon = epsilon

    def get_name(self):
        return 'Cross Entropy'

    def compute(self, predictions: np.ndarray, targets: np.ndarray, model_names: list, metadata_df: pd.DataFrame, actor_name: str, leaderboard_name: str, data_split_name: str, submission_epoch_str: str, output_dirpath: str):
        predictions = predictions.astype(np.float64)
        targets = targets.astype(np.float64)
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        a = targets * np.log(predictions)
        b = (1 - targets) * np.log(1 - predictions)
        ce = -(a + b)

        return {'result': np.average(ce).item(), 'metadata': {'cross_entropy': ce}, 'files': None}

    def compare(self, computed, baseline):
        return computed < baseline

class GroupedCrossEntropyViolin(Metric):
    def __init__(self, write_html:bool = False, share_with_actor:bool = False, store_result_in_submission:bool = False, share_with_external: bool = True, epsilon:float = 1e-12, columns_of_interest: list=None):
        super().__init__(write_html, share_with_actor, store_result_in_submission, share_with_external)
        if columns_of_interest is None:
            self.columns_of_interest = ['all']
        else:
            self.columns_of_interest = columns_of_interest

        if not isinstance(self.columns_of_interest, list):
            raise RuntimeError('Columns of interest must be passed as a list')

        self.epsilon = epsilon

    def get_name(self):
        return 'Grouped Cross Entropy Violin {}'.format('_'.join(self.columns_of_interest))

    def compute(self, predictions: np.ndarray, targets: np.ndarray, model_names: list, metadata_df: pd.DataFrame, actor_name: str, leaderboard_name: str, data_split_name: str, submission_epoch_str: str, output_dirpath: str):
        metadata = {}
        predictions = predictions.astype(np.float64)
        targets = targets.astype(np.float64)
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        a = targets * np.log(predictions)
        b = (1 - targets) * np.log(1 - predictions)
        ce = -(a + b)

        # Identify models based on columns of interest
        model_lists = metadata_utils.build_model_lists(metadata_df, self.columns_of_interest)

        datasets = []
        names = []
        xticks = []
        tick = 1

        for key, model_ids in model_lists.items():
            # Group cross entropies
            ce_for_key = np.zeros(len(model_ids))
            index = 0
            for model_id in model_ids:
                model_index = model_names.index(model_id)
                ce_for_key[index] = ce[model_index]
                index += 1

            metadata[key] = ce_for_key
            names.append(key)
            datasets.append(ce_for_key)
            xticks.append(tick)
            tick+=1

        plt.clf()

        fig, axes = plt.subplots(dpi=300)
        axes.violinplot(dataset=datasets)
        axes.set_title('Cross Entropy Violin Plots for {} in {} for dataset {}'.format(actor_name, leaderboard_name,
                                                                                       data_split_name))
        axes.yaxis.grid(True)
        axes.set_xticks(xticks)
        axes.set_xticklabels(names, rotation=15, ha='right')
        axes.set_ylabel('Cross Entropy')

        filepath = os.path.join(output_dirpath, '{}_{}_{}_{}_{}.png'.format(actor_name, submission_epoch_str, self.get_name(), leaderboard_name, data_split_name))

        plt.savefig(filepath, bbox_inches='tight', dpi=300)

        plt.clf()

        return {'result': None, 'metadata': None, 'files': [filepath]}


class CrossEntropyConfidenceInterval(Metric):
    VALID_LEVELS = [90, 95, 98, 99]

    def __init__(self, write_html: bool = True, share_with_actor: bool = False,
                 store_result_in_submission: bool = True, share_with_external: bool = False, level: int = 95, epsilon: float = 1e-12):
        super().__init__(write_html, share_with_actor, store_result_in_submission, share_with_external)
        self.level = level
        self.epsilon = epsilon

        if self.level not in CrossEntropyConfidenceInterval.VALID_LEVELS:
            raise RuntimeError('Level: {}, must be in {}'.format(self.level, CrossEntropyConfidenceInterval.VALID_LEVELS))

    def get_name(self):
        return 'CE {}% CI'.format(self.level)


    def compute(self, predictions: np.ndarray, targets: np.ndarray, model_names: list, metadata_df: pd.DataFrame, actor_name: str, leaderboard_name: str, data_split_name: str, submission_epoch_str: str, output_dirpath: str):
        predictions = predictions.astype(np.float64)
        targets = targets.astype(np.float64)
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        a = targets * np.log(predictions)
        b = (1 - targets) * np.log(1 - predictions)
        ce = -(a + b)

        # sources https://en.wikipedia.org/wiki/Standard_error
        standard_error = np.std(ce) / np.sqrt(float(len(ce)))
        if self.level == 90:
            ci = standard_error * 1.64
        elif self.level == 95:
            ci = standard_error * 1.96
        elif self.level == 98:
            ci = standard_error * 2.33
        elif self.level == 99:
            ci = standard_error * 2.58
        else:
            raise RuntimeError('Unsupported confidence interval level: {}. Must be in [90, 95, 98, 99]'.format(self.level))
        return {'result': float(ci), 'metadata': None, 'files': None}

class BrierScore(Metric):
    def __init__(self, write_html:bool = True, share_with_actor:bool = False, store_result_in_submission:bool = True, share_with_external: bool = False):
        super().__init__(write_html, share_with_actor, store_result_in_submission, share_with_external)

    def get_name(self):
        return 'Brier Score'

    def compute(self, predictions: np.ndarray, targets: np.ndarray, model_names: list, metadata_df: pd.DataFrame, actor_name: str, leaderboard_name: str, data_split_name: str, submission_epoch_str: str, output_dirpath: str):
        predictions = predictions.astype(np.float64)
        targets = targets.astype(np.float64)

        mse = np.mean(np.square(predictions - targets))
        return {'result': float(mse), 'metadata': None, 'files': None}

    def compare(self, computed, baseline):
        return computed > baseline

class Grouped_ROC_AUC(Metric):
    def __init__(self, write_html: bool = False, share_with_actor: bool = False, store_result_in_submission: bool = True, share_with_external: bool = True, columns_of_interest: list=None):
        super().__init__(write_html, share_with_actor, store_result_in_submission, share_with_external)
        self.columns_of_interest = []
        if columns_of_interest is not None:
            self.columns_of_interest = columns_of_interest

        if not isinstance(self.columns_of_interest, list):
            raise RuntimeError('Columns of interest must be passed as a list')


    def get_name(self):
        if len(self.columns_of_interest) == 0:
            interest_text = ''
        else:
            interest_text = '_'.join(self.columns_of_interest)
        return 'Grouped ROC-AUC-{}'.format(interest_text)

    def compute(self, predictions: np.ndarray, targets: np.ndarray, model_names: list, metadata_df: pd.DataFrame, actor_name: str, leaderboard_name: str, data_split_name: str, submission_epoch_str: str, output_dirpath: str):
        result_data = {}
        files = []

        model_lists = metadata_utils.build_model_lists(metadata_df, self.columns_of_interest)
        thresholds = np.arange(0.0, 1.01, 0.01)

        for key, model_ids in model_lists.items():
            preds_for_key = np.zeros(len(model_ids))
            targets_for_key = np.zeros(len(model_ids))

            index = 0
            for model_id in model_ids:
                model_index = model_names.index(model_id)
                preds_for_key[index] = predictions[model_index]
                targets_for_key[index] = targets[model_index]
                index += 1

            TP_counts = list()
            TN_counts = list()
            FP_counts = list()
            FN_counts = list()
            TPR = list()
            FPR = list()

            nb_condition_positive = np.sum(targets_for_key == 1)
            nb_condition_negative = np.sum(targets_for_key == 0)

            for t in thresholds:
                detections = preds_for_key >= t

                # both detections and targets should be a 1d numpy array
                TP_count = np.sum(np.logical_and(detections == 1, targets_for_key == 1))
                FP_count = np.sum(np.logical_and(detections == 1, targets_for_key == 0))
                FN_count = np.sum(np.logical_and(detections == 0, targets_for_key == 1))
                TN_count = np.sum(np.logical_and(detections == 0, targets_for_key == 0))

                TP_counts.append(TP_count)
                FP_counts.append(FP_count)
                FN_counts.append(FN_count)
                TN_counts.append(TN_count)
                if nb_condition_positive > 0:
                    TPR.append(TP_count / nb_condition_positive)
                else:
                    TPR.append(np.nan)
                if nb_condition_negative > 0:
                    FPR.append(FP_count / nb_condition_negative)
                else:
                    FPR.append(np.nan)

            TP_counts = np.asarray(TP_counts).reshape(-1)
            FP_counts = np.asarray(FP_counts).reshape(-1)
            FN_counts = np.asarray(FN_counts).reshape(-1)
            TN_counts = np.asarray(TN_counts).reshape(-1)
            TPR = np.asarray(TPR).reshape(-1)
            FPR = np.asarray(FPR).reshape(-1)
            thresholds = np.asarray(thresholds).reshape(-1)

            auc_value = auc(FPR, TPR)
            result_data[key] = auc_value


            confusion_matrix_filepath = os.path.join(output_dirpath, '{}_{}-{}-{}-{}-{}.csv'.format(actor_name, submission_epoch_str, leaderboard_name,
                                                                       data_split_name, 'Confusion_Matrix', key))

            fs_utils.write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds,
                                            confusion_matrix_filepath)

            roc_filepath = os.path.join(output_dirpath, '{}_{}-{}-{}-{}-{}.png'.format(actor_name, submission_epoch_str, leaderboard_name,
                                                                       data_split_name, 'ROC', key))

            roc_auc = auc(FPR, TPR)

            plt.clf()
            fig = plt.figure(dpi=300)
            lw = 2
            # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
            plt.plot(FPR, TPR, 'b-', marker='o', markersize=4, linewidth=2)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            legend_str = 'ROC AUC = {:02g}'.format(roc_auc)
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic (ROC) for {} and {}'.format(actor_name, key))
            plt.legend([legend_str], loc='lower right')
            plt.savefig(roc_filepath, bbox_inches='tight', dpi=300)
            plt.clf()

            files.append(confusion_matrix_filepath)
            files.append(roc_filepath)


        filepath = os.path.join(output_dirpath, '{}_{}_{}_{}_{}.json'.format(actor_name, submission_epoch_str, self.get_name(), leaderboard_name, data_split_name))

        with open(filepath, 'w') as fp:
            json.dump(result_data, fp,  indent=2)

        files.append(filepath)

        return {'result': None, 'metadata': None, 'files': files}



class ROC_AUC(Metric):
    def __init__(self, write_html:bool = True, share_with_actor:bool = True, store_result_in_submission:bool = True, share_with_external: bool = False):
        super().__init__(write_html, share_with_actor, store_result_in_submission, share_with_external)

    def get_name(self):
        return 'ROC-AUC'

    def compute(self, predictions: np.ndarray, targets: np.ndarray, model_names: list, metadata_df: pd.DataFrame, actor_name: str, leaderboard_name: str, data_split_name: str, submission_epoch_str: str, output_dirpath: str):
        TP_counts = list()
        TN_counts = list()
        FP_counts = list()
        FN_counts = list()
        TPR = list()
        FPR = list()

        thresholds = np.arange(0.0, 1.01, 0.01)

        nb_condition_positive = np.sum(targets == 1)
        nb_condition_negative = np.sum(targets == 0)

        for t in thresholds:
            detections = predictions >= t

            # both detections and targets should be a 1d numpy array
            TP_count = np.sum(np.logical_and(detections == 1, targets == 1))
            FP_count = np.sum(np.logical_and(detections == 1, targets == 0))
            FN_count = np.sum(np.logical_and(detections == 0, targets == 1))
            TN_count = np.sum(np.logical_and(detections == 0, targets == 0))

            TP_counts.append(TP_count)
            FP_counts.append(FP_count)
            FN_counts.append(FN_count)
            TN_counts.append(TN_count)
            if nb_condition_positive > 0:
                TPR.append(TP_count / nb_condition_positive)
            else:
                TPR.append(np.nan)
            if nb_condition_negative > 0:
                FPR.append(FP_count / nb_condition_negative)
            else:
                FPR.append(np.nan)

        TP_counts = np.asarray(TP_counts).reshape(-1)
        FP_counts = np.asarray(FP_counts).reshape(-1)
        FN_counts = np.asarray(FN_counts).reshape(-1)
        TN_counts = np.asarray(TN_counts).reshape(-1)
        TPR = np.asarray(TPR).reshape(-1)
        FPR = np.asarray(FPR).reshape(-1)
        thresholds = np.asarray(thresholds).reshape(-1)

        confusion_matrix_filepath = os.path.join(output_dirpath, '{}_{}-{}-{}-{}.csv'.format(actor_name, submission_epoch_str, leaderboard_name,
                                                                   data_split_name, 'Confusion_Matrix'))

        fs_utils.write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds,
                                        confusion_matrix_filepath)

        roc_filepath = os.path.join(output_dirpath, '{}_{}-{}-{}-{}.png'.format(actor_name, submission_epoch_str, leaderboard_name,
                                                                   data_split_name, 'ROC'))

        roc_auc = auc(FPR, TPR)

        plt.clf()
        fig = plt.figure(dpi=300)
        lw = 2
        # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
        plt.plot(FPR, TPR, 'b-', marker='o', markersize=4, linewidth=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        legend_str = 'ROC AUC = {:02g}'.format(roc_auc)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) for {}'.format(actor_name))
        plt.legend([legend_str], loc='lower right')
        plt.savefig(roc_filepath, bbox_inches='tight', dpi=300)
        plt.clf()

        return {'result': float(auc(FPR, TPR)), 'metadata': {'tpr': TPR, 'fpr': FPR}, 'files': [confusion_matrix_filepath, roc_filepath]}

    def compare(self, computed, baseline):
        return computed > baseline