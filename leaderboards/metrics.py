# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np
from sklearn.metrics import auc


import os

from leaderboards import fs_utils

class Metric(object):
    def __init__(self, write_html: bool, share_with_actor: bool, store_result_in_submission: bool):
        self.write_html = write_html
        self.share_with_actor = share_with_actor
        self.store_result_in_submission = store_result_in_submission
        self.html_priority = 0

    def get_name(self):
        raise NotImplementedError()

    def compute(self, predictions: np.ndarray, targets: np.ndarray):
        raise NotImplementedError()

    def write_data(self, data, output_dirpath):
        raise NotImplementedError()


class AverageCrossEntropy(Metric):
    def __init__(self, write_html:bool = True, share_with_actor:bool = False, store_result_in_submission:bool = True, epsilon:float = 1e-12):
        super().__init__(write_html, share_with_actor, store_result_in_submission)
        self.epsilon = epsilon

    def get_name(self):
        return 'Cross Entropy'

    def compute(self, predictions: np.ndarray, targets: np.ndarray):
        predictions = predictions.astype(np.float64)
        targets = targets.astype(np.float64)
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        a = targets * np.log(predictions)
        b = (1 - targets) * np.log(1 - predictions)
        ce = -(a + b)

        return {'result': np.average(ce).item(), 'metadata': ce}

class CrossEntropyConfidenceInterval(Metric):
    VALID_LEVELS = [90, 95, 98, 99]

    def __init__(self, write_html: bool = True, share_with_actor: bool = False,
                 store_result_in_submission: bool = True, level: int = 95, epsilon: float = 1e-12):
        super().__init__(write_html, share_with_actor, store_result_in_submission)
        self.level = level
        self.epsilon = epsilon

        if self.level not in CrossEntropyConfidenceInterval.VALID_LEVELS:
            raise RuntimeError('Level: {}, must be in {}'.format(self.level, CrossEntropyConfidenceInterval.VALID_LEVELS))

    def get_name(self):
        return 'CE {}% CI'.format(self.level)

    def compute(self, predictions: np.ndarray, targets: np.ndarray):
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
        return {'result': float(ci), 'metadata': ce}

class BrierScore(Metric):
    def __init__(self, write_html:bool = True, share_with_actor:bool = False, store_result_in_submission:bool = True):
        super().__init__(write_html, share_with_actor, store_result_in_submission)

    def get_name(self):
        return 'Brier Score'

    def compute(self, predictions: np.ndarray, targets: np.ndarray):
        predictions = predictions.astype(np.float64)
        targets = targets.astype(np.float64)

        mse = np.mean(np.square(predictions - targets))
        return {'result': float(mse), 'metadata': None}

class ROC_AUC(Metric):
    def __init__(self, write_html:bool = True, share_with_actor:bool = False, store_result_in_submission:bool = True):
        super().__init__(write_html, share_with_actor, store_result_in_submission)

    def get_name(self):
        return 'ROC-AUC'

    def compute(self, predictions: np.ndarray, targets: np.ndarray):
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

        TPR = np.asarray(TPR).reshape(-1)
        FPR = np.asarray(FPR).reshape(-1)

        return {'result': float(auc(FPR, TPR)), 'metadata': [TPR, FPR]}

    def write_data(self, data, output_dirpath):
        TPR, FPR = data['metadata']
        # TODO: Update?
        # generate_roc_image(fpr, tpr, submission.global_results_dirpath, submission.slurm_job_name)


class ConfusionMatrix(Metric):
    def __init__(self, write_html:bool = False, share_with_actor:bool = True, store_result_in_submission:bool = False):
        super().__init__(write_html, share_with_actor, store_result_in_submission)

    def get_name(self):
        return 'Confusion Matrix'

    def compute(self, predictions: np.ndarray, targets: np.ndarray):
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

        return {'result': None, 'metadata': [TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds]}

    def write_data(self, data, output_dirpath):
        output_filepath = os.path.join(output_dirpath, '{}.json'.format(self.get_name()))
        TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = data['metadata']
        fs_utils.write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds, output_filepath)
        return output_filepath
