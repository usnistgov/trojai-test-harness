import os
import numpy as np
import collections
import traceback
import logging
import sklearn

import ground_truth


def compute_metrics(results_dirpath, ground_truth_dirpath):
    try:
        ground_truth_dict = ground_truth.load_ground_truth(ground_truth_dirpath)
    except:
        msg = 'Unable to load ground truth results: "{}".\n{}'.format(ground_truth_dirpath, traceback.format_exc())
        logging.error(msg)
        raise

	# Dictionary storing results -- key = model name, value = prediction
    results = collections.OrderedDict()

    # loop over each model file trojan prediction is being made for
    logging.info('Looping over ground truth files, computing cross entropy loss.')
    for model_name in ground_truth_dict.keys():
        result_filepath = os.path.join(results_dirpath, model_name + ".txt")

        # Check for result file, if its there we read it in
        if os.path.exists(result_filepath):
            try:
                with open(result_filepath) as file:
                    file_contents = file.readline().strip()
                    result = float(file_contents)
            except:
                # if file parsing fails for any reason, the value is nan
                result = np.nan

            # Check to ensure the result correctly parsed into a float
            if np.isnan(result):
                logging.warning('Failed to parse results for model: "{}" as a float. File contents: "{}" parsed into "{}".'.format(model_name, file_contents, result))
                results[model_name] = np.nan
            else:
                results[model_name] = result
        else:  # If the result file does not exist, then we fill it in with the default answer
            logging.warning('Missing results for model "{}" at "{}".'.format(model_name, result_filepath))
            results[model_name] = np.nan

    # compute cross entropy
    default_result = 0.5
    logging.info('Computing cross entropy between predictions and ground truth.')
    logging.info('Predictions (nan will be replaced with "{}"): "{}"'.format(default_result, results))
    predictions = np.array(list(results.values())).reshape(-1, 1)
    targets = np.array(list(ground_truth_dict.values())).reshape(-1, 1)

    if not np.any(np.isfinite(predictions)):
        logging.warning('Found no parse-able results from container execution.')

    num_missing_predictions = np.count_nonzero(np.isnan(predictions))
    num_total_predictions = predictions.size

    logging.info('Missing results for {}/{} models'.format(num_missing_predictions, num_total_predictions))

    predictions[np.isnan(predictions)] = default_result
    elementwise_cross_entropy = ground_truth.binary_cross_entropy(predictions, targets)
    confidence_interval = ground_truth.cross_entropy_confidence_interval(elementwise_cross_entropy)
    ce = float(np.mean(elementwise_cross_entropy))

    TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = ground_truth.gen_confusion_matrix(targets, predictions)
    # cast to a float so its human readable in the joson
    roc_auc = float(sklearn.metrics.auc(FPR, TPR))

    confusion_output_filename = 'confusion.csv'
    confusion_filepath = os.path.join(results_dirpath, confusion_output_filename)
    ground_truth.write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds, confusion_filepath)

    print('Cross Entropy = {}'.format(ce))
    print('ROC AUC = {}'.format(roc_auc))

    with open(os.path.join(results_dirpath, 'cross_entropy.csv'), 'w') as fh:
        fh.write('{}'.format(ce))
    with open(os.path.join(results_dirpath, 'roc_auc.csv'), 'w') as fh:
        fh.write('{}'.format(roc_auc))

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute TrojAI metrics without the test harness')
    parser.add_argument('--results-dirpath', type=str,
                        help='The filepath to the directory containing the results',
                        required=True)

    parser.add_argument('--ground-truth-dirpath', type=str,
                        help='The filepath to the directory containing the ground truth',
                        required=True)
    args = parser.parse_args()

    compute_metrics(args.results_dirpath, args.ground_truth_dirpath)
