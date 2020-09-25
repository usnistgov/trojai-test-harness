# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np


def elementwise_binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray, epsilon=1e-12) -> np.ndarray:
    predictions = predictions.astype(np.float64)
    targets = targets.astype(np.float64)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    a = targets * np.log(predictions)
    b = (1 - targets) * np.log(1 - predictions)
    ce = -(a + b)
    return ce


def binary_brier_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = predictions.astype(np.float64)
    targets = targets.astype(np.float64)

    mse = np.mean(np.square(predictions - targets))
    return float(mse)


def cross_entropy_confidence_interval(elementwise_cross_entropy: np.ndarray, level: int = 95) -> float:
    # sources https://en.wikipedia.org/wiki/Standard_error
    standard_error = np.std(elementwise_cross_entropy) / np.sqrt(float(len(elementwise_cross_entropy)))
    if level == 90:
        ci = standard_error * 1.64
    elif level == 95:
        ci = standard_error * 1.96
    elif level == 98:
        ci = standard_error * 2.33
    elif level == 99:
        ci = standard_error * 2.58
    else:
        raise RuntimeError('Unsupported confidence interval level: {}. Must be in [90, 95, 98, 99]'.format(level))
    return float(ci)


def confusion_matrix(targets, predictions):
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

    return TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds