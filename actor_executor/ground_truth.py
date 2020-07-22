import os
import numpy as np
import collections
import typing
import logging
import traceback
import shutil
import sklearn.metrics
import csv

from drive_io import DriveIO
from google_drive_file import GoogleDriveFile
import time_utils
from mail_io import TrojaiMail
from submission import Submission


def load_ground_truth(ground_truth_dir: str) -> typing.OrderedDict[str, float]:
    # Dictionary storing ground truth data -- key = model name, value = answer/ground truth
    ground_truth_dict = collections.OrderedDict()

    if os.path.exists(ground_truth_dir):
        for ground_truth_model in os.listdir(ground_truth_dir):

            if not ground_truth_model.startswith('id-'):
                continue

            ground_truth_model_dir = os.path.join(ground_truth_dir, ground_truth_model)

            if not os.path.isdir(ground_truth_model_dir):
                continue

            ground_truth_file = os.path.join(ground_truth_model_dir, "ground_truth.csv")

            if not os.path.exists(ground_truth_file):
                continue

            with open(ground_truth_file) as truth_file:
                file_contents = truth_file.readline().strip()
                ground_truth = float(file_contents)
                ground_truth_dict[ground_truth_model] = ground_truth

    if len(ground_truth_dict) == 0:
        raise RuntimeError('ground_truth_dict length was zero. No ground truth found in "{}"'.format(ground_truth_dir))

    return ground_truth_dict


def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray, epsilon=1e-12) -> float:
    predictions = predictions.astype(np.float64)
    targets = targets.astype(np.float64)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    a = targets * np.log(predictions)
    b = (1 - targets) * np.log(1 - predictions)
    ce = np.mean(-(a + b))
    return float(ce)


def gen_confusion_matrix(targets, predictions):
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
        detections = predictions > t

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


def write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds, confusion_filepath):
    with open(confusion_filepath, 'w', newline='\n') as fh:
        fh.write('Threshold, TP, FP, FN, TN, TPR, FPR\n')
        for i in range(len(thresholds)):
            fh.write('{}, {:d}, {:d}, {:d}, {:d}, {}, {}\n'.format(float(thresholds[i]), int(TP_counts[i]), int(FP_counts[i]), int(FN_counts[i]), int(TN_counts[i]), float(TPR[i]), float(FPR[i])))


# def generate_roc_image(fpr, tpr, png_dirpath, filename_prefix):
#     png_filename = filename_prefix + '.png'
#     png_filepath = os.path.join(png_dirpath, png_filename)
#
#     plt.clf()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.savefig(png_filename)


def process_results(submission: Submission, g_drive: DriveIO, log_file_byte_limit: int) -> None:
    logging.info("Checking results for {}".format(submission.actor.name))

    try:
        ground_truth_dict = load_ground_truth(submission.ground_truth_dirpath)
    except:
        msg = 'Unable to load ground truth results: "{}".\n{}'.format(submission.ground_truth_dirpath, traceback.format_exc())
        logging.error(msg)
        TrojaiMail().send(to='trojai@nist.gov', subject='Unable to Load Ground Truth', message=msg)
        raise

    time_str = time_utils.convert_epoch_to_psudo_iso(submission.execution_epoch)
    errors_filepath = os.path.join(submission.global_results_dirpath, submission.actor.name, time_str, "errors.txt")
    slurm_log_filepath = os.path.join(submission.global_results_dirpath, submission.actor.name, time_str, submission.slurm_output_filename)

    # Test log file truncations
    # while os.path.getsize(slurm_log_filepath) < (1.1 * log_file_byte_limit):
    #     with open(slurm_log_filepath, 'a') as fh:
    #         for i in range(100):
    #             a = np.random.randn(1, 1000)
    #             fh.write(np.array2string(a))

    # truncate log file to N bytes
    # if os.path.exists(slurm_log_filepath):
    #     if (1.01 * os.path.getsize(slurm_log_filepath)) > log_file_byte_limit:  # use 1% buffer
    #         shutil.copyfile(slurm_log_filepath, slurm_log_filepath.replace('.txt', '.orig.txt'))
    #         os.truncate(slurm_log_filepath, log_file_byte_limit)
    #
    #         with open(slurm_log_filepath, 'a') as fh:
    #                 fh.write('\n\n**** Log File Truncated ****\n\n')
    #         shutil.copyfile(slurm_log_filepath, slurm_log_filepath.replace('.txt', '.orig.txt'))
    #
    #         found_executing = False
    #         found_done_executing = False
    #         temp_out = slurm_log_filepath + ".tmp"
    #         with open(slurm_log_filepath, 'r') as slurm_log_file_in, open(temp_out, 'w') as slurm_log_file_out:
    #             for line in slurm_log_filepath:
    #                 if 'Container Execution Complete for team' in line:
    #                     found_done_executing = True
    #
    #                 if not found_executing:
    #                     temp_out.write(line)
    #
    #                 if found_done_executing:
    #                     temp_out.write(line)
    #
    #                 if 'Starting Execution of' in line:
    #                     found_executing = True
    #
    #         os.remove(slurm_log_filepath)
    #         os.rename(temp_out, slurm_log_filepath)

    # start logging to the submission log, in addition to server log
    cur_logging_level = logging.getLogger().getEffectiveLevel()
    # set all individual logging handlers to this level
    for handler in logging.getLogger().handlers:
        handler.setLevel(cur_logging_level)
    # this allows us to set the logger itself to debug without modifying the individual handlers
    logging.getLogger().setLevel(logging.DEBUG)  # this enables the higher level debug to show up for the handler we are about to add

    submission_log_handler = logging.FileHandler(slurm_log_filepath)
    submission_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s"))
    submission_log_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(submission_log_handler)

    try:  # try, finally block to ensure removal of submission logging from the logger utility
        logging.info('**************************************************')
        logging.info('Processing {}: Results'.format(submission.actor.name))
        logging.info('**************************************************')

        # Dictionary storing results -- key = model name, value = prediction
        results = collections.OrderedDict()

        # initialize error strings to empty
        submission.web_display_parse_errors = ""
        submission.web_display_execution_errors = ""

        # loop over each model file trojan prediction is being made for
        logging.info('Looping over ground truth files, computing cross entropy loss.')
        for model_name in ground_truth_dict.keys():
            result_filepath = os.path.join(submission.global_results_dirpath, submission.actor.name, time_str, model_name + ".txt")

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
                    if submission.slurm_queue == 'sts':
                        logging.warning('Failed to parse results for model: "{}" as a float. File contents: "{}" parsed into "{}".'.format(model_name, file_contents, result))
                    if ":Result Parse:" not in submission.web_display_parse_errors:
                        submission.web_display_parse_errors += ":Result Parse:"

                    if submission.slurm_queue == 'sts':
                        logging.warning('Unable to parse results for model "{}".'.format(model_name))
                    results[model_name] = np.nan
                else:
                    results[model_name] = result
            else:  # If the result file does not exist, then we fill it in with the default answer
                logging.warning('Missing results for model "{}" at "{}".'.format(model_name, result_filepath))
                results[model_name] = np.nan

        # Get the actual file that was downloaded for the submission
        logging.info('Loading metatdata from the file actually downloaded and evaluated, in case the file changed between the time the job was submitted and it was executed.')
        orig_file = submission.file

        submission_metadata_filepath = os.path.join(submission.global_results_dirpath, submission.actor.name, time_str, submission.actor.name + ".metadata.json")
        if os.path.exists(submission_metadata_filepath):
            try:
                submission.file = GoogleDriveFile.load_json(submission_metadata_filepath)
                submission.actor.last_file_epoch = submission.file.modified_epoch
                if orig_file.id != submission.file.id:
                    logging.info('Originally Submitted File: "{}"'.format(submission.file))
                    logging.info('Updated Submission with Executed File: "{}"'.format(submission.file))
                else:
                    logging.info('Drive file did not change between original submission and execution.')
            except:
                msg = 'Failed to deserialize file: "{}".\n{}'.format(submission_metadata_filepath, traceback.format_exc())
                logging.error(msg)
                submission.web_display_parse_errors += ":Executed File Update:"
        else:
            msg = 'Executed submission file: "{}" could not be found.\n{}'.format(submission_metadata_filepath, traceback.format_exc())
            logging.error(msg)
            submission.web_display_parse_errors += ":Executed File Update:"

        # compute cross entropy
        default_result = 0.5
        logging.info('Computing cross entropy between predictions and ground truth.')
        if submission.slurm_queue == 'sts':
            logging.info('Predictions (nan will be replaced with "{}"): "{}"'.format(default_result, results))
        predictions = np.array(list(results.values())).reshape(-1,1)
        targets = np.array(list(ground_truth_dict.values())).reshape(-1,1)

        if not np.any(np.isfinite(predictions)):
            logging.warning('Found no parse-able results from container execution.')
            submission.web_display_parse_errors += ":No Results:"

        num_missing_predictions = np.count_nonzero(np.isnan(predictions))
        num_total_predictions = predictions.size

        logging.info('Missing results for {}/{} models'.format(num_missing_predictions, num_total_predictions))

        predictions[np.isnan(predictions)] = default_result
        submission.score = float(binary_cross_entropy(predictions, targets))

        TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = gen_confusion_matrix(targets, predictions)
        submission.roc_auc = sklearn.metrics.auc(FPR, TPR)

        confusion_filepath = os.path.join(submission.global_results_dirpath, submission.actor.name, time_str, submission.confusion_output_filename)
        write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds, confusion_filepath)

        # generate_roc_image(fpr, tpr, submission.global_results_dirpath, submission.slurm_job_name)
        logging.info('Binary Cross Entropy Loss: "{}"'.format(submission.score))
        logging.info('ROC AUC: "{}"'.format(submission.roc_auc))
        if len(targets) < 2:
            logging.info("  ROC Curve undefined for vectors of length: {}".format(len(targets)))

    finally:
        # stop outputting logging to submission log file
        logging.getLogger().removeHandler(submission_log_handler)

        # set the global logging handlers back to its original level
        logging.getLogger().setLevel(cur_logging_level)

    # upload confusion matrix file to drive
    try:
        if os.path.exists(confusion_filepath):
            g_drive.upload_and_share(confusion_filepath, submission.actor.email)
        else:
            logging.error('Failed to find confusion matrix file: {}'.format(confusion_filepath))
            submission.web_display_parse_errors += ":Confusion File Missing:"
    except:
        logging.error('Unable to upload confusion matrix output file: {}'.format(confusion_filepath))
        submission.web_display_parse_errors += ":File Upload:"

    # upload log file to drive
    try:
        if os.path.exists(slurm_log_filepath):
            g_drive.upload_and_share(slurm_log_filepath, submission.actor.email)
        else:
            logging.error('Failed to find slurm output log file: {}'.format(slurm_log_filepath))
            submission.web_display_parse_errors += ":Log File Missing:"
    except:
        logging.error('Unable to upload slurm output log file: {}'.format(slurm_log_filepath))
        submission.web_display_parse_errors += ":File Upload:"

    if os.path.exists(errors_filepath):
        logging.error('Found errors log from job execution.')
        with open(errors_filepath) as f:
            submission.web_display_execution_errors = f.readline().strip()

    # if no errors have been recorded, convert empty string to human readable "None"
    if len(submission.web_display_parse_errors.strip()) == 0:
        submission.web_display_parse_errors = "None"
    if len(submission.web_display_execution_errors.strip()) == 0:
        submission.web_display_execution_errors = "None"

    logging.info('After process_results')
    submission.slurm_job_name = None
    submission.actor.job_status = "None"  # reset job status to enable next submission


