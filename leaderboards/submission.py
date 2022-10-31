# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import os
import time
import typing
import pandas as pd
import fcntl

import collections

import numpy as np
import subprocess
import logging
import traceback
from typing import List

from airium import Airium

from leaderboards.drive_io import DriveIO
from leaderboards.mail_io import TrojaiMail
from leaderboards.google_drive_file import GoogleDriveFile
from leaderboards.actor import Actor, ActorManager
from leaderboards import json_io
from leaderboards import slurm
from leaderboards import time_utils
from leaderboards import fs_utils
from leaderboards.leaderboard import Leaderboard
from leaderboards.trojai_config import TrojaiConfig
from leaderboards import hash_utils

class Submission(object):
    def __init__(self, g_file: GoogleDriveFile, actor: Actor, leaderboard: Leaderboard, data_split_name: str, provenance: str='performer', submission_epoch: int=None, slurm_queue_name: str=None, submission_leaderboard: Leaderboard = None):
        self.g_file = g_file
        self.actor_uuid = actor.uuid
        self.leaderboard_name = leaderboard.name
        self.data_split_name = data_split_name
        self.slurm_queue_name = slurm_queue_name
        if self.slurm_queue_name is None:
            self.slurm_queue_name = leaderboard.get_slurm_queue_name(self.data_split_name)
        self.slurm_nice = leaderboard.get_slurm_nice(self.data_split_name)
        self.metric_results = {}
        self.saved_metric_results = {}
        self.execution_runtime = None
        self.model_execution_runtimes = None
        self.submission_epoch = submission_epoch
        if self.submission_epoch is None:
            self.submission_epoch = time_utils.get_current_epoch()

        self.execution_epoch = None
        self.active_slurm_job_name = None
        self.slurm_output_filename = None
        self.confusion_output_filename = None
        self.web_display_parse_errors = "None"
        self.web_display_execution_errors = "None"
        self.provenance = provenance

        submission_epoch_str = time_utils.convert_epoch_to_psudo_iso(self.submission_epoch)

        if submission_leaderboard is None:
            submission_leaderboard = leaderboard

        # create the directory where submissions are stored
        self.actor_submission_dirpath = os.path.join(submission_leaderboard.submission_dirpath, actor.name, submission_epoch_str)

        if not os.path.isdir(self.actor_submission_dirpath):
            logging.info("Submission directory for " + actor.name + " does not exist, creating ...")
            os.makedirs(self.actor_submission_dirpath)

        # create the directory where results are stored
        self.actor_results_dirpath = os.path.join(leaderboard.get_result_dirpath(self.data_split_name), '{}-submission'.format(submission_leaderboard.name), actor.name, '{}-submission'.format(submission_epoch_str))
        if not os.path.isdir(self.actor_results_dirpath):
            logging.info("Results directory for " + actor.name + " does not exist, creating ...")
            os.makedirs(self.actor_results_dirpath)

        self.execution_results_dirpath = None

    def __str__(self) -> str:
        msg = 'file name: "{}", from actor uuid: "{}"'.format(self.g_file.name, self.actor_uuid)
        return msg

    def get_slurm_job_name(self, actor: Actor):
        return '{}_{}_{}'.format(actor.name, self.leaderboard_name, self.data_split_name)

    def get_submission_hash(self):
        return hash_utils.load_hash(self.get_submission_filepath())

    def is_active_job(self):
        if self.active_slurm_job_name is None:
            return False

        stdout, stderr = slurm.squeue(self.active_slurm_job_name, self.slurm_queue_name)  # raises RuntimeError on failure

        stdoutSplitNL = str(stdout).split("\\n")

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

    def has_errors(self):
        if self.web_display_parse_errors == 'None' and self.web_display_execution_errors == 'None':
            return False
        elif self.web_display_parse_errors == 'None' and (self.web_display_execution_errors == ':Container Parameters (metaparameters):' or self.web_display_execution_errors == ':Container Parameters (metaparameters schema):' or self.web_display_execution_errors == ':Container Parameters (learned parameters):'):
            return False
        else:
            return True

    def check(self, trojai_config: TrojaiConfig, g_drive: DriveIO, actor: Actor, leaderboard: Leaderboard, submission_manager: 'SubmissionManager', log_file_byte_limit: int) -> None:

        if self.active_slurm_job_name is None:
            logging.info('Submission "{}_{}" by team "{}" is not active.'.format(self.leaderboard_name, self.data_split_name, actor.name))
            return

        logging.info('Checking status submission from actor "{}".'.format(actor.name))

        stdout, stderr = slurm.squeue(self.active_slurm_job_name, self.slurm_queue_name)  # raises RuntimeError on failure

        stdoutSplitNL = stdout.decode().split("\n")
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
            self.process_results(trojai_config, actor, leaderboard, g_drive, log_file_byte_limit)

            if leaderboard.is_auto_delete_submission(self.data_split_name):
                # delete the container file to avoid filling up disk space
                submission_filepath = self.get_submission_filepath()
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                if os.path.exists(submission_filepath):
                    os.remove(submission_filepath)
            else:
                if self.has_errors():
                    logging.info('Submission contains errors, so will not auto execute other data splits')
                else:
                    auto_execute_split_names = leaderboard.get_auto_execute_split_names(self.data_split_name)
                    if len(auto_execute_split_names) > 0:
                        # Check to see if we need to launch for split name
                        current_hash = self.get_submission_hash()
                        actor_submissions = submission_manager.get_submissions_by_actor(actor)

                        for auto_execute_split_name in auto_execute_split_names:
                            found_matching_submission = False

                            for submission in actor_submissions:
                                if submission.data_split_name == auto_execute_split_name:
                                    submission_hash = submission.get_submission_hash()
                                    if submission_hash == current_hash:
                                        found_matching_submission = True
                                        break

                            if found_matching_submission:
                                logging.info('Found a matching submission between {} and {}'.format(self.data_split_name, auto_execute_split_name))
                            else:
                                # Did not find matching hash, setting up new submission for auto execute split name
                                new_submission = Submission(self.g_file, actor, leaderboard, auto_execute_split_name, 'auto-{}'.format(auto_execute_split_name), self.submission_epoch)
                                submission_manager.add_submission(actor, new_submission)
                                logging.info('Added submission file name "{}" to manager for email "{}" when auto submitting for {}'.format(new_submission.g_file.name, actor.email, auto_execute_split_name))
                                time.sleep(1)
                                exec_epoch = time_utils.get_current_epoch()
                                new_submission.execute(actor, trojai_config, exec_epoch)
        else:
            logging.warning("Incorrect format for stdout from squeue: {}".format(stdoutSplitNL))

            # attempt to process the result
            self.process_results(trojai_config, actor, leaderboard, g_drive, log_file_byte_limit)

            if leaderboard.is_auto_delete_submission(self.data_split_name):
                # delete the container file to avoid filling up disk space
                submission_filepath = self.get_submission_filepath()
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                if os.path.exists(submission_filepath):
                    os.remove(submission_filepath)

        logging.info("After Check submission: {}".format(self))

    def load_ground_truth(self, leaderboard: Leaderboard) -> typing.OrderedDict[str, float]:
        return leaderboard.load_ground_truth(self.data_split_name)

    def load_results(self, ground_truth_dict: typing.OrderedDict[str, float], print_details=True) -> typing.OrderedDict[str, float]:

        # Dictionary storing results -- key = model name, value = prediction
        results = collections.OrderedDict()

        # loop over each model file trojan prediction is being made for
        if print_details:
            logging.info('Loading results.')
        for model_name in ground_truth_dict.keys():
            result_filepath = os.path.join(self.execution_results_dirpath, model_name + ".txt")

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
                    if self.data_split_name == 'sts':
                        logging.warning(
                            'Failed to parse results for model: "{}" as a float. File contents: "{}" parsed into "{}".'.format(
                                model_name, file_contents, result))
                    if ":Result Parse:" not in self.web_display_parse_errors:
                        self.web_display_parse_errors += ":Result Parse:"

                    results[model_name] = np.nan
                else:
                    results[model_name] = result
            else:  # If the result file does not exist, then we fill it in with the default answer
                if print_details:
                    logging.warning('Missing results for model "{}" at "{}".'.format(model_name, result_filepath))
                results[model_name] = np.nan

        return results

    def process_results(self, trojai_config: TrojaiConfig, actor: Actor, leaderboard: Leaderboard, g_drive: DriveIO, log_file_byte_limit: int, update_actor: bool = True, print_details: bool = True) -> None:
        logging.info("Checking results for {}".format(actor.name))

        info_filepath = os.path.join(self.execution_results_dirpath, Leaderboard.INFO_FILENAME)
        slurm_log_filepath = os.path.join(self.actor_submission_dirpath, self.slurm_output_filename)

        container_output_filename = self.g_file.name + '.out'
        container_output_filepath = os.path.join(self.execution_results_dirpath, container_output_filename)

        epoch_str = time_utils.convert_epoch_to_psudo_iso(self.submission_epoch)
        updated_container_output_filename = '{}_{}.{}'.format(actor.name, epoch_str, container_output_filename)
        updated_container_output_filepath = os.path.join(self.execution_results_dirpath, updated_container_output_filename)
        if os.path.exists(container_output_filepath):
            os.rename(container_output_filepath, updated_container_output_filepath)

        if print_details:
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

        # Create team directory on google drive
        try:
            root_actor_folder_id = g_drive.create_folder('trojai_results_{}'.format(actor.name))
            root_external_folder_id = g_drive.create_folder('trojai_plots_external_{}'.format(actor.name))

            actor_submission_folder_id = g_drive.create_folder('{}_{}'.format(leaderboard.name, self.data_split_name), parent_id=root_actor_folder_id)
            external_actor_submission_folder_id = g_drive.create_folder('{}_{}'.format(leaderboard.name, self.data_split_name), parent_id=root_external_folder_id)

        except:
            logging.error('Failed to create google drive actor directories')
            root_actor_folder_id = None
            root_external_folder_id = None
            actor_submission_folder_id = None
            external_actor_submission_folder_id = None

        try:
            # try, finally block ensures that the duplication of the logging stream to the slurm log file (being sent back to the performers) is removed from the logger utility after the ground truth analysis completes
            logging.info('**************************************************')
            logging.info('Processing {}: Results'.format(actor.name))
            logging.info('**************************************************')

            # initialize error strings to empty
            self.web_display_parse_errors = ""
            self.web_display_execution_errors = ""

            # Get the actual file that was downloaded for the submission
            logging.info('Loading metatdata from the file actually downloaded and evaluated, in case the file changed between the time the job was submitted and it was executed.')
            orig_g_file = self.g_file
            submission_metadata_filepath = os.path.join(self.actor_submission_dirpath, actor.name + ".metadata.json")
            if os.path.exists(submission_metadata_filepath):
                try:
                    self.g_file = GoogleDriveFile.load_json(submission_metadata_filepath)
                    if update_actor:
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

            predictions, targets, models = self.get_predictions_targets_models(leaderboard, print_details=print_details)
            metadata_df = leaderboard.load_metadata_csv_into_df()

            # Subset data
            data_split_metadata = metadata_df[metadata_df['data_split'] == self.data_split_name]

            submission_metrics = leaderboard.get_submission_metrics(self.data_split_name)

            # Compute metrics
            for metric_name, metric in submission_metrics.items():
                self.compute_metric(actor.name, metric, predictions, targets, models, data_split_metadata)

            output_files = []
            external_collab_files = []

            # Gather metric output files
            for metric_name, metric_filepaths in self.saved_metric_results.items():
                if metric_name not in submission_metrics:
                    continue

                metric = submission_metrics[metric_name]
                if metric.share_with_actor:
                    if isinstance(metric_filepaths, list):
                        output_files.extend(metric_filepaths)
                    elif isinstance(metric_filepaths, str):
                        output_files.append(metric_filepaths)
                if metric.share_with_external:
                    if isinstance(metric_filepaths, list):
                        external_collab_files.extend(metric_filepaths)
                    elif isinstance(metric_filepaths, str):
                        external_collab_files.append(metric_filepaths)

            # Upload to actor folder
            for output_file in output_files:
                try:
                    if actor_submission_folder_id is not None and os.path.exists(output_file):
                        g_drive.upload(output_file, folder_id=actor_submission_folder_id)
                    else:
                        logging.error('Unable to upload file: {}'.format(output_file))
                        if ":File Upload:" not in self.web_display_parse_errors:
                            self.web_display_parse_errors += ":File Upload:"
                except:
                    logging.error('Unable to upload file: {}'.format(output_file))
                    if ":File Upload:" not in self.web_display_parse_errors:
                        self.web_display_parse_errors += ":File Upload:"

            # Upload to external folder
            for output_file in external_collab_files:
                try:
                    if external_actor_submission_folder_id is not None and os.path.exists(output_file):
                        g_drive.upload(output_file, folder_id=external_actor_submission_folder_id)
                    else:
                        logging.error('Unable to upload external collab file: {}'.format(output_file))
                except:
                    logging.error('Unable to upload external collab file: {}'.format(output_file))
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
            if print_details:
                # stop outputting logging to submission log file
                logging.getLogger().removeHandler(submission_log_handler)

                # set the global logging handlers back to its original level
                logging.getLogger().setLevel(cur_logging_level)

        # upload log file to drive
        try:
            if actor_submission_folder_id is not None and os.path.exists(slurm_log_filepath):
                g_drive.upload(slurm_log_filepath, folder_id=actor_submission_folder_id)
            else:
                logging.error('Failed to find slurm output log file: {}'.format(slurm_log_filepath))
                self.web_display_parse_errors += ":Log File Missing:"
        except:
            logging.error('Unable to upload slurm output log file: {}'.format(slurm_log_filepath))
            if ":File Upload:" not in self.web_display_parse_errors:
                self.web_display_parse_errors += ":File Upload:"

        # upload container output for sts split only
        try:
            if self.data_split_name == 'sts' or self.data_split_name == 'train':
                if actor_submission_folder_id is not None and os.path.exists(updated_container_output_filepath):
                    g_drive.upload(updated_container_output_filepath, folder_id=actor_submission_folder_id)
                else:
                    logging.error('Failed to find container output file: {}'.format(updated_container_output_filepath))
                    self.web_display_parse_errors += ':Container File Missing:'

        except:
            logging.error('Unable to upload container output file: {}'.format(updated_container_output_filepath))
            self.web_display_parse_errors += ':File Upload(container output):'

        # if no errors have been recorded, convert empty string to human readable "None"
        if len(self.web_display_parse_errors.strip()) == 0:
            self.web_display_parse_errors = "None"
        if len(self.web_display_execution_errors.strip()) == 0:
            self.web_display_execution_errors = "None"

        logging.info('After process_results')
        self.active_slurm_job_name = None

        # Share actor and external folders
        try:
            if root_actor_folder_id is not None:
                g_drive.share(root_actor_folder_id, actor.email)
        except:
            logging.error('Unable to share actor folder with {}'.format(actor.email))

        try:
            if root_external_folder_id is not None:
                for email in trojai_config.summary_metric_email_addresses:
                    g_drive.share(root_external_folder_id, email)
        except:
            logging.error('Unable to share external folders with external emails: {}'.format(trojai_config.summary_metric_email_addresses))

        if update_actor:
            actor.update_job_status(leaderboard.name, self.data_split_name, 'None')

    def get_execute_time_str(self):
        return '{}-execute'.format(time_utils.convert_epoch_to_psudo_iso(self.execution_epoch))

    def compute_metric(self, actor_name, metric, predictions, targets, models, metadata_df, store_results=True):
        metric_output_dirpath = os.path.join(self.execution_results_dirpath)
        epoch_str = time_utils.convert_epoch_to_psudo_iso(self.submission_epoch)

        metric_output = metric.compute(predictions, targets, models, metadata_df, actor_name, self.leaderboard_name, self.data_split_name, epoch_str, metric_output_dirpath)

        if store_results:
            if metric.store_result_in_submission:
                self.metric_results[metric.get_name()] = metric_output['result']

            files = metric_output['files']
            self.saved_metric_results[metric.get_name()] = files

        return metric_output

    def get_predictions_targets_models(self, leaderboard: Leaderboard, print_details: bool = False, update_nan_with_default: bool = True):
        try:
            ground_truth_dict = self.load_ground_truth(leaderboard)
        except:

            msg = 'Unable to load ground truth results: "{}-{}".\n{}'.format(leaderboard.name, self.data_split_name,
                                                                             traceback.format_exc())
            logging.error(msg)
            if print_details:
                TrojaiMail().send(to='trojai@nist.gov', subject='Unable to Load Ground Truth', message=msg)
            raise

        # load the results from disk
        results = self.load_results(ground_truth_dict, print_details)

        default_result = leaderboard.get_default_prediction_result()
        if print_details:
            logging.info('Computing cross entropy between predictions and ground truth.')
            if self.data_split_name == 'sts':
                logging.info('Predictions (nan will be replaced with "{}"): "{}"'.format(default_result, results))

        predictions = np.array(list(results.values())).reshape(-1, 1)
        targets = np.array(list(ground_truth_dict.values())).reshape(-1, 1)

        if not np.any(np.isfinite(predictions)) and print_details:
            logging.warning('Found no parse-able results from container execution.')
            self.web_display_parse_errors += ":No Results:"

        num_missing_predictions = np.count_nonzero(np.isnan(predictions))
        num_total_predictions = predictions.size

        logging.info('Missing results for {}/{} models'.format(num_missing_predictions, num_total_predictions))

        if update_nan_with_default:
            predictions[np.isnan(predictions)] = default_result
        model_list = list(ground_truth_dict.keys())
        return predictions, targets, model_list

    def get_submission_filepath(self):
        return os.path.join(self.actor_submission_dirpath, self.g_file.name)

    def execute(self, actor: Actor, trojai_config: TrojaiConfig, execution_epoch: int, execute_local=False, custom_home_dirpath: str=None, custom_scratch_dirpath: str=None, custom_slurm_options=[], custom_python_env_filepath: str = None) -> None:
        logging.info('Executing submission {} by {}'.format(self.g_file.name, actor.name))
        self.execution_epoch = execution_epoch
        self.execution_results_dirpath = os.path.join(self.actor_results_dirpath, self.get_execute_time_str())

        if not os.path.exists(self.execution_results_dirpath):
            logging.debug('Creating result directory: {}'.format(self.execution_results_dirpath))
            os.makedirs(self.execution_results_dirpath)

        if not os.path.exists(self.actor_submission_dirpath):
            logging.debug('Creating submission directory: {}'.format(self.actor_submission_dirpath))
            os.makedirs(self.actor_submission_dirpath)

        self.active_slurm_job_name = self.get_slurm_job_name(actor)

        slurm_script_filepath = trojai_config.slurm_execute_script_filepath
        task_executor_script_filepath = trojai_config.task_evaluator_script_filepath

        python_executable = trojai_config.python_env
        if custom_python_env_filepath is not None:
            python_executable = custom_python_env_filepath

        test_harness_dirpath = trojai_config.trojai_test_harness_dirpath
        control_slurm_queue = trojai_config.control_slurm_queue_name
        submission_filepath = self.get_submission_filepath()
        trojai_config_filepath = trojai_config.trojai_config_filepath

        cpus_per_task = 30
        if self.slurm_queue_name in trojai_config.vm_cpu_cores_per_partition:
            cpus_per_task = trojai_config.vm_cpu_cores_per_partition[self.slurm_queue_name]

        epoch_str = time_utils.convert_epoch_to_psudo_iso(self.submission_epoch)

        self.slurm_output_filename = '{}.{}_{}.{}.log.txt'.format(self.leaderboard_name, actor.name, epoch_str, self.data_split_name)
        slurm_output_filepath = os.path.join(self.execution_results_dirpath, self.slurm_output_filename)
        # cmd_str_list = [slurm_script_filepath, actor.name, actor.email, submission_filepath, result_dirpath,  trojai_config_filepath, self.leaderboard_name, self.data_split_name, test_harness_dirpath, python_executable, task_executor_script_filepath]
        # cmd_str_list = ['sbatch', '--partition', control_slurm_queue, '--parsable', '--nice={}'.format(self.slurm_nice), '--nodes', '1', '--ntasks-per-node', '1', '--cpus-per-task', '1', ':', '--partition', self.slurm_queue_name, '--nice={}'.format(self.slurm_nice), '--nodes', '1', '--ntasks-per-node', '1', '--cpus-per-task', str(cpus_per_task), '--exclusive', '-J', self.active_slurm_job_name, '--parsable', '-o', slurm_output_filepath, slurm_script_filepath, actor.name, actor.email, submission_filepath, self.execution_results_dirpath, trojai_config_filepath, self.leaderboard_name, self.data_split_name, test_harness_dirpath, python_executable, task_executor_script_filepath]
        cmd_str_list = []
        if execute_local:
            if custom_home_dirpath is None or custom_scratch_dirpath is None:
                raise RuntimeError('Local execution requires user-specified home, scratch, and slurm partition')

            sbatch_control_params = ['sbatch']
            sbatch_vm_params = ['--partition', self.slurm_queue_name, '--nice={}'.format(self.slurm_nice), '--nodes',
                                '1', '--ntasks-per-node', '1', '--cpus-per-task', str(cpus_per_task),
                                '-J', self.active_slurm_job_name, '--parsable', '-o', slurm_output_filepath]
            container_launch_params = [slurm_script_filepath,
                                       "--team-name", actor.name,
                                       "--team-email", actor.email,
                                       "--submission-filepath", submission_filepath,
                                       "--result-dirpath", self.execution_results_dirpath,
                                       "--trojai-config-filepath", trojai_config_filepath,
                                       "--leaderboard-name", self.leaderboard_name,
                                       "--data-split-name", self.data_split_name,
                                       "--trojai-test-harness-dirpath", test_harness_dirpath,
                                       "--python-exec", python_executable,
                                       "--task-executor-filepath", task_executor_script_filepath,
                                       "--is-local",
                                       "--custom-home", custom_home_dirpath,
                                       "--custom-scratch", custom_scratch_dirpath]
        else:
            sbatch_control_params = ['sbatch', '--partition', control_slurm_queue, '--parsable', '--nice={}'.format(self.slurm_nice), '--nodes', '1', '--ntasks-per-node', '1', '--cpus-per-task', '1', ':']
            sbatch_vm_params = ['--partition', self.slurm_queue_name, '--nice={}'.format(self.slurm_nice), '--nodes', '1', '--ntasks-per-node', '1', '--cpus-per-task', str(cpus_per_task), '--exclusive', '-J', self.active_slurm_job_name, '--parsable', '-o', slurm_output_filepath]
            container_launch_params = [slurm_script_filepath,
                                       "--team-name", actor.name,
                                       "--team-email", actor.email,
                                       "--submission-filepath", submission_filepath,
                                       "--result-dirpath", self.execution_results_dirpath,
                                       "--trojai-config-filepath", trojai_config_filepath,
                                       "--leaderboard-name", self.leaderboard_name,
                                       "--data-split-name", self.data_split_name,
                                       "--trojai-test-harness-dirpath", test_harness_dirpath,
                                       "--python-exec", python_executable,
                                       "--task-executor-filepath", task_executor_script_filepath]




        if len(custom_slurm_options) > 0:
            sbatch_vm_params.extend(custom_slurm_options)

        cmd_str_list.extend(sbatch_control_params)
        cmd_str_list.extend(sbatch_vm_params)
        cmd_str_list.extend(container_launch_params)

        logging.info('launching sbatch command: \n{}'.format(' '.join(cmd_str_list)))

        out = subprocess.Popen(cmd_str_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        stdout, stderr = out.communicate()

        # Check if there are no errors reported from sbatch
        if stderr == b'':
            job_id = int(stdout.strip())
            actor.update_job_status(self.leaderboard_name, self.data_split_name, 'Queued')
            actor.update_file_status(self.leaderboard_name, self.data_split_name, 'Ok')
            actor.update_last_submission_epoch(self.leaderboard_name, self.data_split_name, self.submission_epoch)
            actor.update_last_file_epoch(self.leaderboard_name, self.data_split_name, self.g_file.modified_epoch)
            logging.info("Slurm job executed with job id: {}".format(job_id))
        else:
            logging.error("The slurm script: {} resulted in errors {}".format(slurm_script_filepath, stderr.decode()))
            logging.info(stdout.decode())
            self.active_slurm_job_name = None
            self.web_display_execution_errors += ":Slurm Script Error:"

    def get_result_table_row(self, a: Airium, actor: Actor, leaderboard: Leaderboard):
        if self.is_active_job():
            return

        submission_timestr = time_utils.convert_epoch_to_iso(self.submission_epoch)

        # if self.execution_epoch == 0 or self.execution_epoch is None:
        #     execute_timestr = "None"
        # else:
        #     execute_timestr = time_utils.convert_epoch_to_iso(self.execution_epoch)
        if self.g_file.modified_epoch == 0 or self.g_file.modified_epoch is None:
            file_timestr = "None"
        else:
            file_timestr = time_utils.convert_epoch_to_iso(self.g_file.modified_epoch)

        if len(self.web_display_execution_errors.strip()) == 0:
            self.web_display_execution_errors = "None"

        if len(self.web_display_parse_errors.strip()) == 0:
            self.web_display_parse_errors = "None"

        metadata_df = leaderboard.load_metadata_csv_into_df()

        # Subset data
        data_split_metadata = metadata_df[metadata_df['data_split'] == self.data_split_name]

        with a.tr():
            a.td(_t=actor.name)
            submission_metrics = leaderboard.get_submission_metrics(self.data_split_name)
            for metric_name, metric in submission_metrics.items():
                if metric.store_result_in_submission:
                    if metric_name not in self.metric_results.keys():
                        predictions, targets, models = self.get_predictions_targets_models(leaderboard)
                        self.compute_metric(actor.name, metric, predictions, targets, models, data_split_metadata)
                if metric.share_with_actor:
                    if metric_name not in self.saved_metric_results.keys():
                        predictions, targets, models = self.get_predictions_targets_models(leaderboard)
                        self.compute_metric(actor.name, metric, predictions, targets, models, data_split_metadata)

                if metric.write_html:
                    metric_value = self.metric_results[metric_name]
                    if isinstance(metric_value, float):
                        a.td(_t=str(round(metric_value, metric.html_decimal_places)))
                    else:
                        a.td(_t=str(metric_value))

            rounded_execution_time = self.execution_runtime
            if rounded_execution_time is not None:
                rounded_execution_time = round(rounded_execution_time, 2)

            a.td(_t=rounded_execution_time)
            a.td(_t=submission_timestr)
            a.td(_t=file_timestr)
            a.td(_t=self.web_display_parse_errors)
            a.td(_t=self.web_display_execution_errors)

class SubmissionManager(object):
    def __init__(self, leaderboard_name):
        self.leaderboard_name = leaderboard_name
        # keyed on uuid
        self.__submissions = dict()

    def __str__(self):
        msg = ""
        for a, submissions in self.__submissions.items():
            msg = msg + "Actor: {}: \n".format(a)
            for s in submissions:
                msg = msg + "  " + s.__str__() + "\n"
        return msg

    def gather_submissions(self, leaderboard: Leaderboard, data_split_name:str, metric_name: str, metric_criteria: float, actor: Actor) -> List[Submission]:
        execution_submissions = list()

        actor_submissions = self.__submissions[actor.uuid]
        submission_metrics = leaderboard.get_submission_metrics(data_split_name)
        metadata_df = leaderboard.load_metadata_csv_into_df()

        # Subset data
        data_split_metadata = metadata_df[metadata_df['data_split'] == data_split_name]

        for submission in actor_submissions:
            if submission.data_split_name == data_split_name:
                metric = submission_metrics[metric_name]

                if metric_name not in submission.metric_results.keys():
                    predictions, targets, models = submission.get_predictions_targets_models(leaderboard, print_details=False)
                    submission.compute_metric(actor.name, metric, predictions, targets, models, data_split_metadata)

                metric_value = submission.metric_results[metric_name]
                if metric.compare(metric_value, metric_criteria):
                    execution_submissions.append(submission)

        return execution_submissions

    def merge_submissions(self, new_submission_manager: 'SubmissionManager'):
        for uuid, submissions in new_submission_manager.__submissions.items():
            if uuid not in self.__submissions.keys():
                self.__submissions[uuid] = list()

            self.__submissions[uuid].extend(submissions)

    def has_active_submission(self, actor: Actor, data_split_name: str):
        submissions = self.get_submissions_by_actor(actor)
        for submission in submissions:
            if submission.data_split_name == data_split_name:
                if submission.is_active_job():
                    return True
        return False


    def add_submission(self, actor: Actor, submission: Submission) -> None:
        self.get_submissions_by_actor(actor).append(submission)

    def get_submissions_by_actor(self, actor: Actor) -> List[Submission]:
        if actor.uuid not in self.__submissions.keys():
            self.__submissions[actor.uuid] = list()

        return self.__submissions[actor.uuid]

    def get_number_submissions(self) -> int:
        count = 0
        for a in self.__submissions.keys():
            count = count + len(self.__submissions[a])
        return count

    def get_number_actors(self) -> int:
        return len(self.__submissions.keys())

    def save_json_custom(self, filepath: str) -> None:
        json_io.write(filepath, self)

    def save_json(self, leaderboard: Leaderboard) -> None:
        self.save_json_custom(leaderboard.submissions_filepath)

    @staticmethod
    def init_file(filepath: str, leaderboard_name: str) -> None:
        # Create the json file if it does not exist already
        if not os.path.exists(filepath):
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))

            submissions = SubmissionManager(leaderboard_name)
            submissions.save_json_custom(filepath)

    @staticmethod
    def load_json_custom(filepath: str, leaderboard_name: str) -> 'SubmissionManager':
        SubmissionManager.init_file(filepath, leaderboard_name)
        return json_io.read(filepath)

    @staticmethod
    def load_json(leaderboard: Leaderboard) -> 'SubmissionManager':
        return SubmissionManager.load_json_custom(leaderboard.submissions_filepath, leaderboard.name)

    def write_score_table_unique(self, output_dirpath, leaderboard: Leaderboard, actor_manager: ActorManager, data_split_name: str):
        result_filename = 'results-unique-{}-{}.html'.format(leaderboard.name, data_split_name)
        result_filepath = os.path.join(output_dirpath, leaderboard.name, result_filename)
        metadata_df = leaderboard.load_metadata_csv_into_df()

        # Subset data
        data_split_metadata = metadata_df[metadata_df['data_split'] == data_split_name]

        a = Airium()

        valid_submissions = {}

        for actor_uuid, submission_list in self.__submissions.items():
            valid_submissions[actor_uuid] = list()

            for submission in submission_list:
                if submission.data_split_name == data_split_name:
                    valid_submissions[actor_uuid].append(submission)


        with a.div(klass='card-body card-body-cascade pb-0'):
            a.h2(klass='pb-q card-title', _t='Results')
            with a.div(klass='table-responsive'):
                with a.table(id='{}-{}-results'.format(leaderboard.name, data_split_name), klass='table table-striped table-bordered table-sm'):
                    with a.thead():
                        with a.tr():
                            a.th(klass='th-sm', _t='Team')
                            submission_metrics = leaderboard.get_submission_metrics(data_split_name)
                            for metric_name, metric in submission_metrics.items():
                                if metric.write_html:
                                    a.th(klass='th-sm', _t=metric_name)

                            a.th(klass='th-sm', _t='Runtime (s)')
                            a.th(klass='th-sm', _t='Submission Timestamp')
                            a.th(klass='th-sm', _t='File Timestamp')
                            a.th(klass='th-sm', _t='Parsing Errors')
                            a.th(klass='th-sm', _t='Launch Errors')
                    with a.tbody():
                        submission_metrics = leaderboard.get_submission_metrics(data_split_name)
                        evaluation_metric_name = leaderboard.get_evaluation_metric_name(data_split_name)
                        metric = submission_metrics[evaluation_metric_name]

                        for actor_uuid, submissions in valid_submissions.items():
                            best_submission_score = None
                            best_submission = None
                            actor = actor_manager.get_from_uuid(actor_uuid)
                            for s in submissions:
                                if s.is_active_job():
                                    continue

                                if evaluation_metric_name in s.metric_results.keys():
                                    metric_score = s.metric_results[evaluation_metric_name]
                                else:
                                    predictions, targets, models = s.get_predictions_targets_models(leaderboard)
                                    s.compute_metric(actor.name, metric, predictions, targets, models, data_split_metadata)
                                    metric_score = s.metric_results[evaluation_metric_name]

                                if best_submission_score is None or metric.compare(metric_score, best_submission_score):
                                    best_submission_score = metric_score
                                    best_submission = s

                            if best_submission is not None:
                                best_submission.get_result_table_row(a, actor, leaderboard)

        with open(result_filepath, 'w') as f:
            f.write(str(a))

        return result_filepath

    def write_score_table(self, output_dirpath, leaderboard: Leaderboard, actor_manager: ActorManager, data_split_name: str):
        result_filename = 'results-{}-{}.html'.format(leaderboard.name, data_split_name)
        result_filepath = os.path.join(output_dirpath, leaderboard.name, result_filename)
        a = Airium()

        valid_submissions = {}

        for actor_uuid, submission_list in self.__submissions.items():
            valid_submissions[actor_uuid] = list()

            for submission in submission_list:
                if submission.data_split_name == data_split_name:
                    valid_submissions[actor_uuid].append(submission)

        with a.div(klass='card-body card-body-cascade pb-0'):
            a.h2(klass='pb-q card-title', _t='All Results')
            with a.div(klass='table-responsive'):
                with a.table(id='{}-{}-all-results'.format(leaderboard.name, data_split_name),
                             klass='table table-striped table-bordered table-sm'):
                    with a.thead():
                        with a.tr():
                            a.th(klass='th-sm', _t='Team')
                            submission_metrics = leaderboard.get_submission_metrics(data_split_name)
                            for metric_name, metric in submission_metrics.items():
                                if metric.write_html:
                                    a.th(klass='th-sm', _t=metric_name)

                            a.th(klass='th-sm', _t='Runtime (s)')
                            a.th(klass='th-sm', _t='Submission Timestamp')
                            a.th(klass='th-sm', _t='File Timestamp')
                            a.th(klass='th-sm', _t='Parsing Errors')
                            a.th(klass='th-sm', _t='Launch Errors')
                    with a.tbody():
                        for actor_uuid, submissions in valid_submissions.items():
                            actor = actor_manager.get_from_uuid(actor_uuid)
                            for s in submissions:
                                s.get_result_table_row(a, actor, leaderboard)

        with open(result_filepath, 'w') as f:
            f.write(str(a))

        return result_filepath

    def generate_round_results_csv(self, leaderboard: Leaderboard, actor_manager: ActorManager, overwrite_csv: bool = True):

        df = leaderboard.load_metadata_csv_into_df()

        if os.path.exists(leaderboard.summary_results_csv_filepath) and not overwrite_csv:
            logging.info('Skipping building round results: {} already exists and overwrite is disabled.'.format(leaderboard.summary_results_csv_filepath))
            return

        all_dfs = []

        default_result = leaderboard.get_default_prediction_result()
        metadata_df = leaderboard.load_metadata_csv_into_df()

        for actor in actor_manager.get_actors():
            submissions = self.get_submissions_by_actor(actor)
            for data_split in leaderboard.get_all_data_split_names():
                leaderboard_metrics = leaderboard.get_submission_metrics(data_split)

                all_model_ids = list(df.loc[df['data_split'] == data_split, 'model_name'].unique())

                metrics = {}
                for submission in submissions:
                    if submission.is_active_job():
                        continue

                    if submission.data_split_name == data_split:
                        raw_predictions_np, raw_targets_np, model_names = submission.get_predictions_targets_models(leaderboard, update_nan_with_default=False, print_details=False)
                        predictions_np = np.copy(raw_predictions_np)
                        predictions_np[np.isnan(predictions_np)] = default_result

                        # Subset data
                        data_split_metadata = metadata_df[metadata_df['data_split'] == data_split]
                        time_str = time_utils.convert_epoch_to_iso(submission.submission_epoch)

                        # Get full metric results
                        for metric_name, metric in leaderboard_metrics.items():
                            metric_output = metric.compute(predictions_np, raw_targets_np, model_names, data_split_metadata, actor.name, leaderboard.name, data_split, time_str, submission.execution_results_dirpath)

                            metadata = metric_output['metadata']

                            if metadata is not None:
                                if isinstance(metadata, dict):
                                    for key, value in metadata.items():
                                        metrics[key] = value
                                else:
                                    raise RuntimeError('Unexpected type for metadata: {}'.format(metadata))
                        new_data = {}
                        new_data['model_name'] = all_model_ids
                        new_data['team_name'] = [actor.name] * len(all_model_ids)
                        new_data['submission_timestamp'] = [time_str] * len(all_model_ids)
                        new_data['data_split'] = [data_split] * len(all_model_ids)
                        new_data['prediction'] = [float(i) for i in raw_predictions_np]
                        new_data['ground_truth'] = [float(i) for i in raw_targets_np]
                        for key, value in metrics.items():
                            data = [float(i) for i in value]
                            if len(data) == len(all_model_ids):
                                new_data[key] = data

                        new_df = pd.DataFrame(new_data)
                        all_dfs.append(new_df)

        result_df = pd.concat(all_dfs)

        result_df.to_csv(leaderboard.summary_results_csv_filepath, index=False)

        logging.info('Finished writing round results to {}'.format(leaderboard.summary_results_csv_filepath))

        return result_df

    def recompute_metrics(self, trojai_config: TrojaiConfig, leaderboard: Leaderboard):

        actor_manager = ActorManager.load_json(trojai_config)
        g_drive = DriveIO(trojai_config.token_pickle_filepath)
        log_file_byte_limit = trojai_config.log_file_byte_limit

        for actor_uuid, submissions in self.__submissions.items():
            actor = actor_manager.get_from_uuid(actor_uuid)
            for submission in submissions:
                # Verify it is not active prior to computing metrics
                if not submission.is_active_job():
                    submission.process_results(trojai_config, actor, leaderboard, g_drive, log_file_byte_limit, update_actor=False, print_details=False)

        self.save_json(leaderboard)


def merge_submissions(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    leaderboard = Leaderboard.load_json(trojai_config, args.name)
    new_submission_manager = SubmissionManager.load_json_custom(args.new_submissions_filepath, leaderboard.name)
    submission_manager = SubmissionManager.load_json(leaderboard)

    num_submissions_prior = submission_manager.get_number_submissions()

    submission_manager.merge_submissions(new_submission_manager)
    submission_manager.save_json(leaderboard)

    submissions_after_merge = submission_manager.get_number_submissions()
    print('Finished merging, new submissions added: {}'.format(submissions_after_merge-num_submissions_prior))


def recompute_metrics(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

    print('Attempting to acquire PID file lock.')
    lock_file = '/var/lock/trojai-lockfile'
    if args.unsafe:
        leaderboard = Leaderboard.load_json(trojai_config, args.name)
        submission_manager = SubmissionManager.load_json(leaderboard)
        submission_manager.recompute_metrics(trojai_config, leaderboard)
        print('Finished recomputing metrics for {}'.format(leaderboard.name))
    else:
        with open(lock_file, 'w') as f:
            try:
                fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print('  PID lock acquired')
                leaderboard = Leaderboard.load_json(trojai_config, args.name)
                submission_manager = SubmissionManager.load_json(leaderboard)
                submission_manager.recompute_metrics(trojai_config, leaderboard)
                print('Finished recomputing metrics for {}'.format(leaderboard.name))
            except OSError as e:
                print('check-and-launch was already running when called. {}'.format(e))
            finally:
                fcntl.lockf(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Runs leaderboards commands')
    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    merge_submissions_parser = subparser.add_parser('merge-submissions')
    merge_submissions_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    merge_submissions_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    merge_submissions_parser.add_argument('--new-submissions-filepath', type=str, help='The filepath to the new submissions to merge into the leaderboard', required=True)
    merge_submissions_parser.set_defaults(func=merge_submissions)

    recompute_metrics_parser = subparser.add_parser('recompute-metrics')
    recompute_metrics_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    recompute_metrics_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    recompute_metrics_parser.add_argument('--unsafe', action='store_true', help='Disables trojai lock (useful for debugging only)')
    recompute_metrics_parser.set_defaults(func=recompute_metrics)

    args = parser.parse_args()
    args.func(args)
