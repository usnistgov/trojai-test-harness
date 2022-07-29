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

from trojai_leaderboard.drive_io import DriveIO
from trojai_leaderboard.mail_io import TrojaiMail
from trojai_leaderboard.google_drive_file import GoogleDriveFile
from trojai_leaderboard.actor import Actor
from trojai_leaderboard import json_io
from trojai_leaderboard import slurm
from trojai_leaderboard import time_utils
from trojai_leaderboard import fs_utils
from trojai_leaderboard.leaderboard import Leaderboard
from trojai_leaderboard.trojai_config import TrojaiConfig


class Submission(object):
    def __init__(self, g_file: GoogleDriveFile, actor: Actor, leaderboard: Leaderboard, data_split_name: str):
        self.g_file = g_file
        self.actor_name = actor.name
        self.actor_email = actor.email
        self.leaderboard_name = leaderboard.name
        self.data_split_name = data_split_name
        self.slurm_queue_name = leaderboard.get_slurm_queue_name(self.data_split_name)
        self.metric_results = {}
        self.saved_metric_results = {}
        self.execution_runtime = None
        self.model_execution_runtimes = None
        self.execution_epoch = None
        self.active_slurm_job_name = None
        self.slurm_output_filename = None
        self.confusion_output_filename = None
        self.web_display_parse_errors = "None"
        self.web_display_execution_errors = "None"

        self.ground_truth_dirpath = leaderboard.get_ground_truth_dirpath(self.data_split_name)

        # create the directory where submissions are stored
        self.actor_submission_dirpath = os.path.join(leaderboard.submission_dirpath, self.actor_name)

        if not os.path.isdir(self.actor_submission_dirpath):
            logging.info("Submission directory for " + self.actor_name + " does not exist, creating ...")
            os.makedirs(self.actor_submission_dirpath)

        # create the directory where results are stored
        self.actor_results_dirpath = os.path.join(leaderboard.get_result_dirpath(self.data_split_name), leaderboard.name, self.actor_name)
        if not os.path.isdir(self.actor_results_dirpath):
            logging.info("Results directory for " + self.actor_name + " does not exist, creating ...")
            os.makedirs(self.actor_results_dirpath)

    def __str__(self) -> str:
        msg = 'file name: "{}", from email: "{}"'.format(self.g_file.name, self.actor_email)
        return msg

    def get_slurm_job_name(self):
        return '{}_{}_{}'.format(self.actor_name, self.leaderboard_name, self.data_split_name)

    def is_active_job(self):
        if self.active_slurm_job_name is None:
            return False

        stdout, stderr = slurm.squeue(self.active_slurm_job_name, self.slurm_queue_name)  # raises RuntimeError on failure

        stdoutSplitNL = str(stdout).split("\\n")
        logging.info('squeue results: {}'.format(stdoutSplitNL))

        # Check if we got a valid response from squeue
        if len(stdoutSplitNL) == 3:
            # found single job with that name, and it has state
            info = stdoutSplitNL[1]
            info_split = info.strip().split(' ')
            # slurm_status = str(info_split[0]).strip()

            if len(info_split) != 1:
                logging.warning("Incorrect format for status info: {}".format(info_split))

            return True

        elif len(stdoutSplitNL) == 2:
            return False
        else:
            logging.warning("Incorrect format for stdout from squeue: {}".format(stdoutSplitNL))
            return False

    def check(self, g_drive: DriveIO, actor: Actor, leaderboard: Leaderboard, submission_manager: 'SubmissionManager', log_file_byte_limit: int) -> None:

        if self.active_slurm_job_name is None:
            logging.info('Submission "{}_{}" by team "{}" is not active.'.format(self.leaderboard_name, self.data_split_name, self.actor_name))
            return

        logging.info('Checking status submission from actor "{}".'.format(self.actor_name))

        stdout, stderr = slurm.squeue(self.active_slurm_job_name, self.slurm_queue_name)  # raises RuntimeError on failure

        stdoutSplitNL = str(stdout).split("\\n")
        logging.info('squeue results: {}'.format(stdoutSplitNL))

        # Check if we got a valid response from squeue
        if len(stdoutSplitNL) == 3:
            # found single job with that name, and it has state
            info = stdoutSplitNL[1]
            info_split = info.strip().split(' ')
            slurm_status = str(info_split[0]).strip()
            logging.info('slurm has status: {} for job name: {}'.format(slurm_status, self.active_slurm_job_name))
            if len(info_split) == 1:
                actor.update_job_status(self.leaderboard_name, self.data_split_name, slurm_status)
            else:
                logging.warning("Incorrect format for status info: {}".format(info_split))
        elif len(stdoutSplitNL) == 2:
            logging.info('squeue does not have status for job name: {}'.format(self.active_slurm_job_name))
            # 1 entries means no state and job name was not found
            # if the job was not found, and this was a previously active submission, the results are ready for processing
            self.process_results(actor, leaderboard, g_drive, log_file_byte_limit)

            if self.data_split_name == 'sts':
                # delete the container file to avoid filling up disk space for the STS server
                time_str = time_utils.convert_epoch_to_psudo_iso(self.execution_epoch)
                submission_filepath = os.path.join(self.actor_submission_dirpath, time_str, self.g_file.name)
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                os.remove(submission_filepath)
            elif self.data_split_name == 'test':
                # TODO: Implement checking for train results/submission . . .
                # If it does not exist, then create a new submission using the test container
                # use submission_manager to find valid submissions ... alternatively could use the actor_submission list...

                pass
            elif self.data_split_name == 'train':
                # TODO: Check for matching test container
                pass

        else:
            logging.warning("Incorrect format for stdout from squeue: {}".format(stdoutSplitNL))

            # attempt to process the result
            self.process_results(actor, leaderboard, g_drive, log_file_byte_limit)

            if self.data_split_name == 'sts':
                # delete the container file to avoid filling up disk space for the STS server
                time_str = time_utils.convert_epoch_to_psudo_iso(self.execution_epoch)
                submission_filepath = os.path.join(self.actor_submission_dirpath, time_str, self.g_file.name)
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                os.remove(submission_filepath)
            elif self.data_split_name == 'test':
                # TODO: Implement checking for train results/submission . . .
                # If it does not exist, then create a new submission using the test container
                # use submission_manager to find valid submissions ... alternatively could use the actor_submission list...

                pass
            elif self.data_split_name == 'train':
                # TODO: Check for matching test container
                pass

        logging.info("After Check submission: {}".format(self))

    def process_results(self, actor: Actor, leaderboard: Leaderboard, g_drive: DriveIO, log_file_byte_limit: int) -> None:
        logging.info("Checking results for {}".format(self.actor_name))

        time_str = time_utils.convert_epoch_to_psudo_iso(self.execution_epoch)
        info_filepath = os.path.join(self.actor_submission_dirpath, time_str, Leaderboard.INFO_FILENAME)
        slurm_log_filepath = os.path.join(self.actor_submission_dirpath, time_str, self.slurm_output_filename)

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
            logging.info('Processing {}: Results'.format(self.actor_name))
            logging.info('**************************************************')

            # initialize error strings to empty
            self.web_display_parse_errors = ""
            self.web_display_execution_errors = ""

            # Get the actual file that was downloaded for the submission
            logging.info('Loading metatdata from the file actually downloaded and evaluated, in case the file changed between the time the job was submitted and it was executed.')
            orig_g_file = self.g_file
            submission_metadata_filepath = os.path.join(self.actor_submission_dirpath, time_str, self.actor_name + ".metadata.json")
            if os.path.exists(submission_metadata_filepath):
                try:
                    self.g_file = GoogleDriveFile.load_json(submission_metadata_filepath)
                    actor.update_last_file_epoch(leaderboard.name, self.data_split_name, self.g_file.modified_epoch)

                    if orig_g_file.id != self.g_file.id:
                        logging.info('Originally Submitted File: "{}, id: {}"'.format(orig_g_file.name, orig_g_file.id))
                        logging.info('Updated Submission with Executed File: "{}"'.format(self.g_file))
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
            if self.data_split_name == 'sts':
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


            metric_output_dirpath = os.path.join(self.actor_submission_dirpath, time_str)

            # Compute metrics
            for metric in leaderboard.get_submission_metrics(self.data_split_name):
                metric_name = metric.get_name()
                metric_output = metric.compute(predictions, targets)

                if metric.store_result_in_submission:
                    self.metric_results[metric_name] = metric_output['result']

                if metric.share_with_actor:
                    self.saved_metric_results[metric_name] = metric.write_data(metric_output, metric_output_dirpath)

            output_files = []

            # Gather metric output files
            for metric_filepaths in self.saved_metric_results.values():
                if isinstance(metric_filepaths, list):
                    output_files.extend(metric_filepaths)
                elif isinstance(metric_filepaths, str):
                    output_files.append(metric_filepaths)

            # Share metric output files with actor
            for output_file in output_files:
                try:
                    if os.path.exists(output_file):
                        g_drive.upload_and_share(output_file, self.actor_email)
                    else:
                        logging.error('Unable to upload file: {}'.format(output_file))
                        if ":File Upload:" not in self.web_display_parse_errors:
                            self.web_display_parse_errors += ":File Upload:"
                except:
                    logging.error('Unable to upload file: {}'.format(output_file))
                    if ":File Upload:" not in self.web_display_parse_errors:
                        self.web_display_parse_errors += ":File Upload:"

            # load the runtime info from the vm-executor
            if not os.path.exists(info_filepath):
                logging.error('Failed to find vm-executor info json dictionary file: {}'.format(info_filepath))
                self.web_display_parse_errors += ":Info File Missing:"
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

        # upload log file to drive
        try:
            if os.path.exists(slurm_log_filepath):
                g_drive.upload_and_share(slurm_log_filepath, self.actor_email)
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
        self.active_slurm_job_name = None

        actor.update_job_status(leaderboard.name, self.data_split_name, 'None')


    def execute(self, actor: Actor, slurm_queue: str, trojai_config: TrojaiConfig, execution_epoch: int) -> None:
        logging.info('Executing submission {} by {}'.format(self.g_file.name, self.actor_name))

        time_str = time_utils.convert_epoch_to_psudo_iso(execution_epoch)

        result_dirpath = os.path.join(self.actor_results_dirpath, time_str)
        if not os.path.exists(result_dirpath):
            logging.debug('Creating result directory: {}'.format(result_dirpath))
            os.makedirs(result_dirpath)

        submission_dirpath = os.path.join(self.actor_submission_dirpath, time_str)
        if not os.path.exists(submission_dirpath):
            logging.debug('Creating submission directory: {}'.format(submission_dirpath))
            os.makedirs(submission_dirpath)

        self.active_slurm_job_name = self.get_slurm_job_name()

        slurm_script_filepath = trojai_config.slurm_execute_script_filepath
        v100_slurm_queue = trojai_config.control_slurm_queue_name
        submission_filepath = os.path.join(submission_dirpath, self.g_file.name)

        # TODO: Update run_python.sh
        # New version should indicate the following:
        # 1. The filepath to the Leaderboard (used to fetch the task)
        # 2. The submission filepath
        # 3. The results dirpath
        # 4. The team name
        # 5. The email for the team

        # select which slurm queue to use and build the command string list
        # if self.slurm_queue == 'sts':
        #     self.slurm_output_filename = self.actor.name + ".sts.log.txt"
        #     self.confusion_output_filename = self.actor.name + ".sts.confusion.csv"
        #     slurm_output_filepath = os.path.join(result_dirpath, self.slurm_output_filename)
        #
        #     cmd_str_list = ['sbatch', "--partition", v100_slurm_queue, "--nodes", "1", "--ntasks-per-node", "1", "--cpus-per-task", "1", ":", "--partition", self.slurm_queue, "--nodes", "1", "--ntasks-per-node", "1", "--cpus-per-task", "10", "--gres=gpu:1", "-J", self.slurm_job_name, "--parsable", "-o", slurm_output_filepath, slurm_script, self.actor.name, submission_dirpath, result_dirpath, config_filepath, self.actor.email, slurm_output_filepath]
        # else:
        #     self.slurm_output_filename = self.actor.name + ".es.log.txt"
        #     self.confusion_output_filename = self.actor.name + ".es.confusion.csv"
        #     slurm_output_filepath = os.path.join(result_dirpath, self.slurm_output_filename)
        #
        #     # ES queue uses "--nice" option to reduce priority by 100, allowing the STS to run when the ES if full
        #     cmd_str_list = ['sbatch', "--partition", v100_slurm_queue, "--nodes", "1", "--ntasks-per-node", "1", "--cpus-per-task", "1", ":", "--partition", self.slurm_queue, "--nodes", "1", "--ntasks-per-node", "1", "--cpus-per-task", "30", "--gres=gpu:3", "-J", self.slurm_job_name, "--nice", "--parsable", "-o", slurm_output_filepath, slurm_script, self.actor.name, submission_dirpath, result_dirpath, config_filepath, self.actor.email, slurm_output_filepath]
        cmd_str_list = ['placeholder']
        logging.info('launching sbatch command: \n{}'.format(' '.join(cmd_str_list)))
        out = subprocess.Popen(cmd_str_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = out.communicate()

        # Check if there are no errors reported from sbatch
        if stderr == b'':
            job_id = int(stdout.strip())
            self.execution_epoch = execution_epoch
            actor.update_job_status(self.leaderboard_name, self.data_split_name, 'Queued')
            actor.update_file_status(self.leaderboard_name, self.data_split_name, 'Ok')
            actor.update_last_execution_epoch(self.leaderboard_name, self.data_split_name, execution_epoch)
            actor.update_last_file_epoch(self.leaderboard_name, self.data_split_name, self.g_file.modified_epoch)
            logging.info("Slurm job executed with job id: {}".format(job_id))
        else:
            logging.error("The slurm script: {} resulted in errors {}".format(slurm_script_filepath, stderr))
            self.web_display_execution_errors += ":Slurm Script Error:"


class SubmissionManager(object):
    def __init__(self, leaderboard_name):
        self.leaderboard_name = leaderboard_name
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

    def has_active_submission(self, actor: Actor):
        submissions = self.__submissions[actor.email]
        for submission in submissions:
            if submission.is_active_job():
                return True
        return False


    def add_submission(self, actor: Actor, submission: Submission) -> None:
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
        json_io.write(filepath, self)

    @staticmethod
    def init_file(filepath: str, leaderboard_name: str) -> None:
        # Create the json file if it does not exist already
        if not os.path.exists(filepath):
            submissions = SubmissionManager(leaderboard_name)
            submissions.save_json(filepath)

    @staticmethod
    def load_json(filepath: str, leaderboard_name: str):
        SubmissionManager.init_file(filepath, leaderboard_name)
        return json_io.read(filepath)

    def get_score_table_unique(self):

            # TODO: Update
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
            # TODO: Update
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
