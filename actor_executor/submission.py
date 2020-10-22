# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import subprocess
import logging
import traceback
from typing import List
from typing import Dict
import sklearn.metrics

from actor_executor.drive_io import DriveIO
from actor_executor.mail_io import TrojaiMail
from actor_executor.google_drive_file import GoogleDriveFile
from actor_executor.actor import Actor
from actor_executor import json_io
from actor_executor import slurm
from actor_executor import time_utils
from actor_executor import fs_utils
from actor_executor import metrics


class Submission(object):
    def __init__(self, gdrive_file: GoogleDriveFile, actor: Actor, submission_dirpath: str, results_dirpath: str, ground_truth_dirpath: str, slurm_queue: str):
        self.file = gdrive_file
        self.actor = actor

        self.slurm_queue = slurm_queue
        self.cross_entropy = None
        self.cross_entropy_95_confidence_interval = None
        self.roc_auc = None
        self.brier_score = None
        self.execution_runtime = None
        self.model_execution_runtimes = None
        self.execution_epoch = None
        self.slurm_job_name = None
        self.slurm_output_filename = None
        self.confusion_output_filename = None
        self.web_display_parse_errors = "None"
        self.web_display_execution_errors = "None"

        self.ground_truth_dirpath = ground_truth_dirpath

        # create the directory where submissions are stored
        self.global_submission_dirpath = submission_dirpath
        if not os.path.isdir(os.path.join(self.global_submission_dirpath, self.actor.name)):
            logging.info("Submission directory for " + self.actor.name + " does not exist, creating ...")
            os.makedirs(os.path.join(self.global_submission_dirpath, self.actor.name))

        # create the directory where results are stored
        self.global_results_dirpath = results_dirpath
        if not os.path.isdir(os.path.join(self.global_results_dirpath, self.actor.name)):
            logging.info("Results directory for " + self.actor.name + " does not exist, creating ...")
            os.makedirs(os.path.join(self.global_results_dirpath, self.actor.name))

    def __str__(self) -> str:
        msg = 'file name: "{}", from eamil: "{}"'.format(self.file.name, self.actor.email)
        return msg

    def check(self, g_drive: DriveIO, log_file_byte_limit: int) -> None:

        if self.slurm_job_name is None:
            logging.info('Submission "{}" by team "{}" is not active.'.format(self.file.name, self.actor.name))
            return

        logging.info('Checking status submission from actor "{}".'.format(self.actor.name))
        stdout, stderr = slurm.squeue(self.slurm_job_name, self.slurm_queue)  # raises RuntimeError on failure

        stdoutSplitNL = str(stdout).split("\\n")
        logging.info('squeue results: {}'.format(stdoutSplitNL))

        # Check if we got a valid response from squeue
        if len(stdoutSplitNL) == 3:
            # found single job with that name, and it has state
            info = stdoutSplitNL[1]
            info_split = info.strip().split(' ')
            slurm_status = str(info_split[0]).strip()
            logging.info('slurm has status: {} for job name: {}'.format(slurm_status, self.slurm_job_name))
            if len(info_split) == 1:
                self.actor.job_status = slurm_status
            else:
                logging.warning("Incorrect format for status info: {}".format(info_split))
        elif len(stdoutSplitNL) == 2:
            logging.info('squeue does not have status for job name: {}'.format(self.slurm_job_name))
            # 1 entries means no state and job name was not found
            # if the job was not found, and this was a previously active submission, the results are ready for processing
            self.process_results(g_drive, log_file_byte_limit)

            if self.slurm_queue == 'sts':
                # delete the container file to avoid filling up disk space for the STS server
                time_str = time_utils.convert_epoch_to_psudo_iso(self.execution_epoch)
                submission_filepath = os.path.join(self.global_submission_dirpath, self.actor.name, time_str, self.file.name)
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                os.remove(submission_filepath)
        else:
            logging.warning("Incorrect format for stdout from squeue: {}".format(stdoutSplitNL))

            # attempt to process the result
            self.process_results(g_drive, log_file_byte_limit)

            if self.slurm_queue == 'sts':
                # delete the container file to avoid filling up disk space for the STS server
                time_str = time_utils.convert_epoch_to_psudo_iso(self.execution_epoch)
                submission_filepath = os.path.join(self.global_submission_dirpath, self.actor.name, time_str, self.file.name)
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                os.remove(submission_filepath)

        logging.info("After Check submission: {}".format(self))

    def execute(self, slurm_script, config_filepath: str, execution_epoch: int) -> None:
        logging.info('Executing submission {} by {}'.format(self.file.name, self.actor.name))

        time_str = time_utils.convert_epoch_to_psudo_iso(execution_epoch)

        result_dirpath = os.path.join(self.global_results_dirpath, self.actor.name, time_str)
        if not os.path.exists(result_dirpath):
            logging.debug('Creating result directory: {}'.format(result_dirpath))
            os.makedirs(result_dirpath)

        submission_dirpath = os.path.join(self.global_submission_dirpath, self.actor.name, time_str)
        if not os.path.exists(submission_dirpath):
            logging.debug('Creating submission directory: {}'.format(submission_dirpath))
            os.makedirs(submission_dirpath)

        # select which slurm queue to use
        if self.slurm_queue == 'sts':
            self.slurm_output_filename = self.actor.name + ".sts.log.txt"
            self.confusion_output_filename = self.actor.name + ".sts.confusion.csv"
            slurm_output_filepath = os.path.join(result_dirpath, self.slurm_output_filename)
        else:
            self.slurm_output_filename = self.actor.name + ".es.log.txt"
            self.confusion_output_filename = self.actor.name + ".es.confusion.csv"
            slurm_output_filepath = os.path.join(result_dirpath, self.slurm_output_filename)

        self.slurm_job_name = self.actor.name
        v100_slurm_queue = 'control'
        cmd_str_list = ['sbatch', "--partition", v100_slurm_queue, "-n", "1", ":", "--partition", self.slurm_queue, "--gres=gpu:1", "-J", self.slurm_job_name,"--parsable", "-o", slurm_output_filepath, slurm_script, self.actor.name, submission_dirpath, result_dirpath, config_filepath, self.actor.email, slurm_output_filepath]
        logging.info('launching sbatch command: "{}"'.format(' '.join(cmd_str_list)))
        out = subprocess.Popen(cmd_str_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = out.communicate()

        # Check if there are no errors reported from sbatch
        if stderr == b'':
            job_id = int(stdout.strip())
            self.execution_epoch = execution_epoch

            self.actor.job_status = "Queued"
            self.actor.file_status = "Ok"
            self.actor.last_execution_epoch = execution_epoch
            self.actor.last_file_epoch = self.file.modified_epoch
            logging.info("Slurm job executed with job id: {}".format(job_id))
        else:
            logging.error("The slurm script: {} resulted in errors {}".format(slurm_script, stderr))
            self.web_display_execution_errors += ":Slurm Script Error:"

    def process_results(self, g_drive: DriveIO, log_file_byte_limit: int) -> None:
        logging.info("Checking results for {}".format(self.actor.name))

        time_str = time_utils.convert_epoch_to_psudo_iso(self.execution_epoch)
        info_filepath = os.path.join(self.global_results_dirpath, self.actor.name, time_str, "info.json")
        slurm_log_filepath = os.path.join(self.global_results_dirpath, self.actor.name, time_str, self.slurm_output_filename)

        # truncate log file to N bytes
        fs_utils.truncate_log_file(slurm_log_filepath, log_file_byte_limit)

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

        try:
            # try, finally block ensures that the duplication of the logging stream to the slurm log file (being sent back to the performers) is removed from the logger utility after the ground truth analysis completes
            logging.info('**************************************************')
            logging.info('Processing {}: Results'.format(self.actor.name))
            logging.info('**************************************************')

            # initialize error strings to empty
            self.web_display_parse_errors = ""
            self.web_display_execution_errors = ""

            # Get the actual file that was downloaded for the submission
            logging.info('Loading metatdata from the file actually downloaded and evaluated, in case the file changed between the time the job was submitted and it was executed.')
            orig_file = self.file

            submission_metadata_filepath = os.path.join(self.global_results_dirpath, self.actor.name, time_str, self.actor.name + ".metadata.json")
            if os.path.exists(submission_metadata_filepath):
                try:
                    self.file = GoogleDriveFile.load_json(submission_metadata_filepath)
                    self.actor.last_file_epoch = self.file.modified_epoch
                    if orig_file.id != self.file.id:
                        logging.info('Originally Submitted File: "{}"'.format(self.file))
                        logging.info('Updated Submission with Executed File: "{}"'.format(self.file))
                    else:
                        logging.info('Drive file did not change between original submission and execution.')
                except:
                    msg = 'Failed to deserialize file: "{}".\n{}'.format(submission_metadata_filepath, traceback.format_exc())
                    logging.error(msg)
                    self.web_display_parse_errors += ":Executed File Update:"
            else:
                msg = 'Executed submission file: "{}" could not be found.\n{}'.format(submission_metadata_filepath, traceback.format_exc())
                logging.error(msg)
                self.web_display_parse_errors += ":Executed File Update:"

            try:
                ground_truth_dict = fs_utils.load_ground_truth(self.ground_truth_dirpath)
            except:
                msg = 'Unable to load ground truth results: "{}".\n{}'.format(self.ground_truth_dirpath, traceback.format_exc())
                logging.error(msg)
                TrojaiMail().send(to='trojai@nist.gov', subject='Unable to Load Ground Truth', message=msg)
                raise

            # load the results from disk
            results = fs_utils.load_results(ground_truth_dict, self, time_str)

            # compute cross entropy
            default_result = 0.5
            logging.info('Computing cross entropy between predictions and ground truth.')
            if self.slurm_queue == 'sts':
                logging.info('Predictions (nan will be replaced with "{}"): "{}"'.format(default_result, results))
            predictions = np.array(list(results.values())).reshape(-1, 1)
            targets = np.array(list(ground_truth_dict.values())).reshape(-1, 1)

            if not np.any(np.isfinite(predictions)):
                logging.warning('Found no parse-able results from container execution.')
                self.web_display_parse_errors += ":No Results:"

            num_missing_predictions = np.count_nonzero(np.isnan(predictions))
            num_total_predictions = predictions.size

            logging.info('Missing results for {}/{} models'.format(num_missing_predictions, num_total_predictions))

            predictions[np.isnan(predictions)] = default_result
            elementwise_cross_entropy = metrics.elementwise_binary_cross_entropy(predictions, targets)
            ce_95_ci = metrics.cross_entropy_confidence_interval(elementwise_cross_entropy)
            self.cross_entropy = float(np.mean(elementwise_cross_entropy))
            self.cross_entropy_95_confidence_interval = ce_95_ci

            # compute Brier score
            self.brier_score = metrics.binary_brier_score(predictions, targets)

            TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = metrics.confusion_matrix(targets, predictions)
            # cast to a float so its human readable in the joson
            self.roc_auc = float(sklearn.metrics.auc(FPR, TPR))

            confusion_filepath = os.path.join(self.global_results_dirpath, self.actor.name, time_str, self.confusion_output_filename)
            fs_utils.write_confusion_matrix(TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds, confusion_filepath)

            # generate_roc_image(fpr, tpr, submission.global_results_dirpath, submission.slurm_job_name)
            logging.info('Binary Cross Entropy Loss: "{}"'.format(self.cross_entropy))
            logging.info('ROC AUC: "{}"'.format(self.roc_auc))
            if len(targets) < 2:
                logging.info("  ROC Curve undefined for vectors of length: {}".format(len(targets)))
            logging.info('Brier Score: "{}"'.format(self.brier_score))

            # load the runtime info from the vm-executor
            if not os.path.exists(info_filepath):
                logging.error('Failed to find vm-executor info json dictionary file: {}'.format(info_filepath))
                self.web_display_parse_errors += ":Info File Missing:"
                # TODO add ":Info File Missing:" to the web display of errors
            else:
                info_dict = json_io.read(info_filepath)
                if 'execution_runtime' not in info_dict.keys():
                    logging.error("Missing 'execution_runtime' key in info file dictionary")
                    self.execution_runtime = np.nan
                else:
                    self.execution_runtime = info_dict['execution_runtime']

                if 'model_execution_runtimes' not in info_dict.keys():
                    self.model_execution_runtimes = dict()
                else:
                    self.model_execution_runtimes = info_dict['model_execution_runtimes']

                if 'errors' not in info_dict.keys():
                    logging.error("Missing 'errors' key in info file dictionary")
                else:
                    self.web_display_execution_errors = info_dict['errors']

        finally:
            # stop outputting logging to submission log file
            logging.getLogger().removeHandler(submission_log_handler)

            # set the global logging handlers back to its original level
            logging.getLogger().setLevel(cur_logging_level)

        # upload confusion matrix file to drive
        try:
            if os.path.exists(confusion_filepath):
                g_drive.upload_and_share(confusion_filepath, self.actor.email)
            else:
                logging.error('Failed to find confusion matrix file: {}'.format(confusion_filepath))
                self.web_display_parse_errors += ":Confusion File Missing:"
        except:
            logging.error('Unable to upload confusion matrix output file: {}'.format(confusion_filepath))
            if ":File Upload:" not in self.web_display_parse_errors:
                self.web_display_parse_errors += ":File Upload:"

        # upload log file to drive
        try:
            if os.path.exists(slurm_log_filepath):
                g_drive.upload_and_share(slurm_log_filepath, self.actor.email)
            else:
                logging.error('Failed to find slurm output log file: {}'.format(slurm_log_filepath))
                self.web_display_parse_errors += ":Log File Missing:"
        except:
            logging.error('Unable to upload slurm output log file: {}'.format(slurm_log_filepath))
            if ":File Upload:" not in self.web_display_parse_errors:
                self.web_display_parse_errors += ":File Upload:"

        # if no errors have been recorded, convert empty string to human readable "None"
        if len(self.web_display_parse_errors.strip()) == 0:
            self.web_display_parse_errors = "None"
        if len(self.web_display_execution_errors.strip()) == 0:
            self.web_display_execution_errors = "None"

        logging.info('After process_results')
        self.slurm_job_name = None
        self.actor.job_status = "None"  # reset job status to enable next submission


class SubmissionManager(object):
    def __init__(self):
        # keyed on email
        self.__submissions = dict()

    def __str__(self):
        msg = ""
        for a in self.__submissions.keys():
            msg = msg + "Actor: {}: \n".format(a)
            submissions = self.__submissions[a]
            for s in submissions:
                msg = msg + "  " + s.__str__() + "\n"
        return msg

    def gather_submissions(self, min_loss_criteria: float, execute_team_name: str) -> Dict[str, List[Submission]]:
        holdout_execution_submissions = dict()

        for actor_email in self.__submissions.keys():
            submissions = self.__submissions[actor_email]
            accepted_submissions = list()

            for submission in submissions:
                if execute_team_name is not None and execute_team_name != submission.actor.name:
                    break

                if submission.cross_entropy is not None and submission.cross_entropy < min_loss_criteria:
                    accepted_submissions.append(submission)

            if len(accepted_submissions) > 0:
                holdout_execution_submissions[actor_email] = accepted_submissions

        return holdout_execution_submissions

    def add_submission(self, submission: Submission) -> None:
        actor = submission.actor
        if actor.email not in self.__submissions.keys():
            self.__submissions[actor.email] = list()
        self.__submissions[actor.email].append(submission)

    def get_submissions_by_actor(self, actor: Actor) -> List[Submission]:
        if actor.email in self.__submissions.keys():
            return self.__submissions[actor.email]
        else:
            return list()

    def get_number_submissions(self) -> int:
        count = 0
        for a in self.__submissions.keys():
            count = count + len(self.__submissions[a])
        return count

    def get_number_actors(self) -> int:
        return len(self.__submissions.keys())

    def save_json(self, filepath: str) -> None:
        # make copies of all the actors to ensure json file is human readable on disk
        import copy
        for actor_email in self.__submissions.keys():
            submissions = self.__submissions[actor_email]
            for submission in submissions:
                submission.actor = copy.deepcopy(submission.actor)
        json_io.write(filepath, self)

    @staticmethod
    def init_file(filepath: str) -> None:
        # Create the json file if it does not exist already
        if not os.path.exists(filepath):
            submissions = SubmissionManager()
            submissions.save_json(filepath)

    @staticmethod
    def load_json(filepath: str):
        SubmissionManager.init_file(filepath)
        return json_io.read(filepath)

    def get_score_table_unique(self):
        # ["Team", "Cross Entropy", "CE 95% CI", "Brier Score", "ROC-AUC", "Runtime (s)", "Execution Timestamp", "File Timestamp", "Parsing Errors", "Launch Errors"]
        scores = []

        for key in self.__submissions.keys():
            submissions = self.__submissions[key]
            best_submission_score = 9999
            best_submission = None
            for s in submissions:
                if s.execution_epoch == 0 or s.execution_epoch is None:
                    execute_timestr = "None"
                else:
                    execute_timestr = time_utils.convert_epoch_to_iso(s.execution_epoch)
                if s.file.modified_epoch == 0 or s.file.modified_epoch is None:
                    file_timestr = "None"
                else:
                    file_timestr = time_utils.convert_epoch_to_iso(s.file.modified_epoch)

                if len(s.web_display_execution_errors.strip()) == 0:
                    s.web_display_execution_errors = "None"

                if len(s.web_display_parse_errors.strip()) == 0:
                    s.web_display_parse_errors = "None"

                if s.cross_entropy is not None:
                    if best_submission_score > s.cross_entropy:
                        best_submission_score = s.cross_entropy
                        best_submission = [s.actor.name, s.cross_entropy, s.cross_entropy_95_confidence_interval,
                                           s.brier_score, s.roc_auc, s.execution_runtime, execute_timestr, file_timestr,
                                           s.web_display_parse_errors, s.web_display_execution_errors]
            if best_submission is not None:
                scores.append(best_submission)
        return scores

    def get_score_table(self):
        # ["Team", "Cross Entropy", "CE 95% CI", "Brier Score", "ROC-AUC", "Runtime (s)", "Execution Timestamp", "File Timestamp", "Parsing Errors", "Launch Errors"]
        scores = []

        for key in self.__submissions.keys():
            submissions = self.__submissions[key]
            for s in submissions:
                if s.execution_epoch == 0 or s.execution_epoch is None:
                    execute_timestr = "None"
                else:
                    execute_timestr = time_utils.convert_epoch_to_iso(s.execution_epoch)
                if s.file.modified_epoch == 0 or s.file.modified_epoch is None:
                    file_timestr = "None"
                else:
                    file_timestr = time_utils.convert_epoch_to_iso(s.file.modified_epoch)

                if len(s.web_display_execution_errors.strip()) == 0:
                    s.web_display_execution_errors = "None"

                if len(s.web_display_parse_errors.strip()) == 0:
                    s.web_display_parse_errors = "None"

                if s.cross_entropy is not None:
                    scores.append(
                        [s.actor.name, s.cross_entropy, s.cross_entropy_95_confidence_interval, s.brier_score,
                         s.roc_auc, s.execution_runtime, execute_timestr, file_timestr, s.web_display_parse_errors,
                         s.web_display_execution_errors])
        return scores