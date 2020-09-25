# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import collections
import typing
import logging
import shutil

from actor_executor.submission import Submission


def truncate_log_file(filepath: str, byte_limit: int):
    # truncate log file to N bytes
    if os.path.exists(filepath) and (byte_limit is not None) and (byte_limit > 0):
        if (1.01 * os.path.getsize(filepath)) > byte_limit:  # use 1% buffer
            shutil.copyfile(filepath, filepath.replace('.txt', '.orig.txt'))
            os.truncate(filepath, byte_limit)

            with open(filepath, 'a') as fh:
                fh.write('\n\n**** Log File Truncated ****\n\n')


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


def write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds, confusion_filepath):
    with open(confusion_filepath, 'w', newline='\n') as fh:
        fh.write('Threshold, TP, FP, FN, TN, TPR, FPR\n')
        for i in range(len(thresholds)):
            fh.write('{}, {:d}, {:d}, {:d}, {:d}, {}, {}\n'.format(float(thresholds[i]), int(TP_counts[i]), int(FP_counts[i]), int(FN_counts[i]), int(TN_counts[i]), float(TPR[i]), float(FPR[i])))


def load_results(ground_truth_dict: typing.OrderedDict[str, float], submission: Submission, time_str: str):
    # Dictionary storing results -- key = model name, value = prediction
    results = collections.OrderedDict()

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

                results[model_name] = np.nan
            else:
                results[model_name] = result
        else:  # If the result file does not exist, then we fill it in with the default answer
            logging.warning('Missing results for model "{}" at "{}".'.format(model_name, result_filepath))
            results[model_name] = np.nan

    return results