# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
import json
import math
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import os
import time
import pandas as pd
import fcntl


import numpy as np
import subprocess
import logging
import traceback
from typing import List, Dict
from datetime import datetime
from airium import Airium
from python_utils import update_object_values, get_value

from leaderboards.drive_io import DriveIO
from leaderboards.google_drive_file import GoogleDriveFile
from leaderboards.actor import Actor, ActorManager
from leaderboards import json_io
from leaderboards import slurm
from leaderboards import time_utils
from leaderboards.leaderboard import *
from leaderboards.trojai_config import TrojaiConfig
from leaderboards import hash_utils
from leaderboards import jsonschema_checker
from leaderboards.results_manager import ResultsManager

class Submission(object):
    def __init__(self, g_file: GoogleDriveFile, actor: Actor, leaderboard: Leaderboard, data_split_name: str, provenance: str='performer', submission_epoch: int=None, slurm_queue_name: str=None, submission_leaderboard: Leaderboard = None):
        self.g_file = g_file
        self.actor_uuid = actor.uuid
        self.leaderboard_name = leaderboard.name
        self.leaderboard_revision = leaderboard.revision
        self.data_split_name = data_split_name
        self.slurm_queue_name = slurm_queue_name
        if self.slurm_queue_name is None:
            self.slurm_queue_name = leaderboard.get_slurm_queue_name(self.data_split_name)
        self.slurm_nice = leaderboard.get_slurm_nice(self.data_split_name)
        self.execution_runtime = None
        self.model_execution_runtimes = None
        self.submission_epoch = submission_epoch
        if self.submission_epoch is None:
            self.submission_epoch = time_utils.get_current_epoch()

        self.execution_epoch = None
        self.active_slurm_job_name = None
        self.slurm_output_filename = None
        self.web_display_parse_errors = "None"
        self.web_display_execution_errors = "None"
        self.provenance = provenance
        self.processed_metric_names = []
        #self.metric_results
        #self.saved_metric_results

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
        submission_epoch_str = time_utils.convert_epoch_to_psudo_iso(self.submission_epoch)
        return '{}_{}_{}_{}'.format(actor.name, self.leaderboard_name, self.data_split_name, submission_epoch_str)

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

    def has_errors(self, actor: Actor):
        if self.web_display_parse_errors == 'None' and self.web_display_execution_errors == 'None':
            return False
        elif self.web_display_parse_errors == 'None':
            if actor.type == 'performer':
                return True

            exec_error_split = self.web_display_execution_errors.strip().split(':')
            exec_error_split = list(filter(None, exec_error_split))
            for error_msg in exec_error_split:
                if 'Container Parameters' not in error_msg or 'Schema Header' not in error_msg:
                    return True
            return False
        else:
            return True

    def check(self, trojai_config: TrojaiConfig, results_manager: ResultsManager, g_drive: DriveIO, actor: Actor, leaderboard: Leaderboard, submission_manager: 'SubmissionManager', log_file_byte_limit: int) -> None:

        if self.active_slurm_job_name is None:
            logging.info('Submission "{}_{}" by team "{}" is not active.'.format(self.leaderboard_name, self.data_split_name, actor.name))
            return

        logging.info('Checking status submission from actor "{}".'.format(actor.name))

        stdout, stderr = slurm.squeue(self.active_slurm_job_name, self.slurm_queue_name)  # raises RuntimeError on failure

        stdoutSplitNL = stdout.decode().split("\n")
        logging.info('squeue results: {}'.format(stdoutSplitNL))

        # Check if we got a valid response from squeue
        # Check slurm output if job is in queue
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

        # Job did not have status in queue
        elif len(stdoutSplitNL) == 2:
            logging.info('squeue does not have status for job name: {}'.format(self.active_slurm_job_name))
            # 1 entries means no state and job name was not found
            # if the job was not found, and this was a previously active submission, the results are ready for processing
            self.process_completed_submission(trojai_config, results_manager, actor, leaderboard, g_drive, log_file_byte_limit)

            if leaderboard.is_auto_delete_submission(self.data_split_name):
                # delete the container file to avoid filling up disk space
                submission_filepath = self.get_submission_filepath()
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                if os.path.exists(submission_filepath):
                    os.remove(submission_filepath)
            else:
                # If the submission had errors, then we should not attempt to run the submission beyond what we have done
                if self.has_errors(actor):
                    logging.info('Submission contains errors, so will not auto execute other data splits')
                else:
                    # If no errors then we see if this submission is supposed to run on any other data splits
                    auto_execute_split_names = leaderboard.get_auto_execute_split_names(self.data_split_name)
                    if len(auto_execute_split_names) > 0:
                        # Check to see if we need to launch for split name
                        current_hash = self.get_submission_hash()
                        actor_submissions = submission_manager.get_submissions_by_actor(actor)

                        # Loop through the auto execute submissions and see if the submission for that data split exists and if the hash is the same
                        # If they are the same, then we should not relaunch the submission
                        for auto_execute_split_name in auto_execute_split_names:
                            found_matching_submission = False

                            for submission in actor_submissions:
                                if submission.data_split_name == auto_execute_split_name:
                                    submission_hash = submission.get_submission_hash()
                                    if submission_hash == current_hash:
                                        found_matching_submission = True
                                        break

                            # Skip submissions that we found have matching submissions (already ran)
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
        # Unknown format coming from slurm, attempt to process results for whatever is there
        else:
            logging.warning("Incorrect format for stdout from squeue: {}".format(stdoutSplitNL))

            # attempt to process the result
            self.process_completed_submission(trojai_config, results_manager, actor, leaderboard, g_drive, log_file_byte_limit)

            if leaderboard.is_auto_delete_submission(self.data_split_name):
                # delete the container file to avoid filling up disk space
                submission_filepath = self.get_submission_filepath()
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                if os.path.exists(submission_filepath):
                    os.remove(submission_filepath)

        logging.info("After Check submission: {}".format(self))

    def dump_summary_schema_csv(self, trojai_config: TrojaiConfig, actor_name: str,  leaderboard: Leaderboard):
        summary_schema_csv_filepath = leaderboard.get_summary_schema_csv_filepath(trojai_config)

        default_schema_keys = ['$schema', 'title', 'technique', 'technique_description', 'technique_changes', 'commit_id', 'repo_name', 'required', 'additionalProperties', 'type', 'properties']

        submission_filepath = self.get_submission_filepath()
        if not os.path.exists(submission_filepath):
            logging.info('The submission no longer exists {}'.format(submission_filepath))
            return

        schema_dict = jsonschema_checker.collect_json_metaparams_schema(submission_filepath)

        new_csv = False
        if not os.path.exists(summary_schema_csv_filepath):
            new_csv = True

        with open(summary_schema_csv_filepath, 'a') as f:
            if new_csv:
                f.write('team_name,data_split,{},submission_filepath'.format(','.join(default_schema_keys)))

            schema_output = '{},{},'.format(actor_name, self.data_split_name)

            for schema_key in default_schema_keys:
                if schema_dict is None or schema_key not in schema_dict:
                    schema_output += 'None,'
                else:
                    schema_output += '{},'.format(schema_dict[schema_key])

            schema_output += '{}\n'.format(submission_filepath)

            f.write(schema_output)

    def process_completed_submission(self, trojai_config: TrojaiConfig, results_manager: ResultsManager, actor: Actor, leaderboard: Leaderboard, g_drive: DriveIO, log_file_byte_limit: int, update_actor: bool = True, print_details: bool = True, output_metaparams_csv: bool = True):
        logging.info('Processing completed submission for {}'.format(actor.name))

        ##################################################
        # initialize error strings to empty
        ##################################################
        self.web_display_parse_errors = ""
        self.web_display_execution_errors = ""

        ##################################################
        # Fetch (or create) team directory on google drive
        ##################################################
        actor_submission_folder_id, external_actor_submission_folder_id = g_drive.get_submission_actor_and_external_folder_ids(actor.name, leaderboard.name, self.data_split_name)

        ##################################################
        # Dumps submissions metadata file into CSV format
        ##################################################
        if output_metaparams_csv:
            self.dump_summary_schema_csv(trojai_config, actor.name, leaderboard)

        info_filepath = os.path.join(self.execution_results_dirpath, Leaderboard.INFO_FILENAME)
        slurm_log_filepath = os.path.join(self.execution_results_dirpath, self.slurm_output_filename)

        ##################################################
        # Renames container output file to associate with actor and submission epoch
        ##################################################
        container_output_filename = os.path.splitext(self.g_file.name)[0] + '.out'
        container_output_filepath = os.path.join(self.execution_results_dirpath, container_output_filename)
        epoch_str = time_utils.convert_epoch_to_psudo_iso(self.submission_epoch)
        updated_container_output_filename = '{}_{}.{}'.format(actor.name, epoch_str, container_output_filename + '.txt')
        updated_container_output_filepath = os.path.join(self.execution_results_dirpath, updated_container_output_filename)

        if os.path.exists(container_output_filepath):
            os.rename(container_output_filepath, updated_container_output_filepath)
        else:
            logging.warning("Missing output log file: {}".format(container_output_filepath))

        ##################################################
        # Check if file stored when submission was first sent is different from shared file
        # This could represent a change in the processed file during slurm queuing time, so we check what the submission
        # Actually used compared to what was submitted
        ##################################################
        logging.info(
            'Checking metatdata from the file actually downloaded and evaluated, in case the file changed between the time the job was submitted and it was executed.')
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
                msg = 'Failed to deserialize file: "{}".\n{}'.format(submission_metadata_filepath,
                                                                     traceback.format_exc())
                logging.error(msg)
                self.web_display_parse_errors += ":Executed File Update:"
        else:
            msg = 'Executed submission file: "{}" could not be found.\n{}'.format(submission_metadata_filepath,
                                                                                  traceback.format_exc())
            logging.error(msg)
            self.web_display_parse_errors += ":Executed File Update:"

        ##################################################
        # load the runtime info from the task and process metadata about execution
        ##################################################
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

                # Check for early abort to reset actor time window
                if 'Container Parameters' in self.web_display_execution_errors or 'Schema Header' in self.web_display_execution_errors:
                    if actor.type == 'performer':
                        actor.reset_leaderboard_time_window(leaderboard.name, self.data_split_name)

        ##################################################
        # Process the metrics from the submission
        ##################################################
        error_dict, processed_metric_names = leaderboard.process_metrics(g_drive, results_manager, self.data_split_name, self.execution_results_dirpath, actor.name, actor.uuid, self.get_submission_epoch_str_primary(), self.processed_metric_names)

        self.processed_metric_names.extend(processed_metric_names)

        if 'web_display_parse_errors' in error_dict:
            self.web_display_parse_errors += error_dict['web_display_parse_errors']


        ##################################################
        # upload log file to drive
        ##################################################
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

        ##################################################
        # upload container output for sts split only
        ##################################################
        try:
            if self.data_split_name == 'sts' or self.data_split_name == 'train':
                if actor_submission_folder_id is not None and os.path.exists(updated_container_output_filepath):
                    g_drive.upload(updated_container_output_filepath, folder_id=actor_submission_folder_id)
                else:
                    logging.error(
                        'Failed to find container output file: {}'.format(updated_container_output_filepath))
                    self.web_display_parse_errors += ':Container File Missing:'

        except:
            logging.error('Unable to upload container output file: {}'.format(updated_container_output_filepath))
            self.web_display_parse_errors += ':File Upload(container output):'

        ##################################################
        # if no errors have been recorded, convert empty string to human readable "None"
        ##################################################
        if len(self.web_display_parse_errors.strip()) == 0:
            self.web_display_parse_errors = "None"
        if len(self.web_display_execution_errors.strip()) == 0:
            self.web_display_execution_errors = "None"

        ##################################################
        # Finished processing, reset actor status and mark no active slurm job for submission
        ##################################################
        self.active_slurm_job_name = None

        if update_actor:
            actor.update_job_status(leaderboard.name, self.data_split_name, 'None')

        logging.info('After process_results')

    def get_execute_time_str(self):
        return '{}-execute'.format(time_utils.convert_epoch_to_psudo_iso(self.execution_epoch))

    def get_submission_filepath(self):
        return os.path.join(self.actor_submission_dirpath, self.g_file.name)

    def get_submission_epoch_str_primary(self):
        return time_utils.convert_epoch_to_iso(self.submission_epoch)

    def execute(self, actor: Actor, trojai_config: TrojaiConfig, execution_epoch: int, execute_local=False, custom_home_dirpath: str=None, custom_scratch_dirpath: str=None, custom_slurm_options=None, custom_python_env_filepath: str = None) -> None:
        if custom_slurm_options is None:
            custom_slurm_options = []

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

        self.slurm_output_filename = '{}.{}_{}_{}.log.txt'.format(self.leaderboard_name, actor.name, epoch_str, self.data_split_name)
        slurm_output_filepath = os.path.join(self.execution_results_dirpath, self.slurm_output_filename)
        # cmd_str_list = [slurm_script_filepath, actor.name, actor.email, submission_filepath, result_dirpath,  trojai_config_filepath, self.leaderboard_name, self.data_split_name, test_harness_dirpath, python_executable, task_executor_script_filepath]
        # cmd_str_list = ['sbatch', '--partition', control_slurm_queue, '--parsable', '--nice={}'.format(self.slurm_nice), '--nodes', '1', '--ntasks-per-node', '1', '--cpus-per-task', '1', ':', '--partition', self.slurm_queue_name, '--nice={}'.format(self.slurm_nice), '--nodes', '1', '--ntasks-per-node', '1', '--cpus-per-task', str(cpus_per_task), '--exclusive', '-J', self.active_slurm_job_name, '--parsable', '-o', slurm_output_filepath, slurm_script_filepath, actor.name, actor.email, submission_filepath, self.execution_results_dirpath, trojai_config_filepath, self.leaderboard_name, self.data_split_name, test_harness_dirpath, python_executable, task_executor_script_filepath]
        cmd_str_list = []
        if execute_local:
            if custom_home_dirpath is None or custom_scratch_dirpath is None:
                raise RuntimeError('Local execution requires user-specified home, scratch, and slurm partition')

            sbatch_control_params = ['sbatch']
            sbatch_vm_params = ['--partition', self.slurm_queue_name, 
            '--nice={}'.format(self.slurm_nice), 
            '--nodes', '1', 
            '--ntasks-per-node', '1', 
            '--cpus-per-task', str(cpus_per_task), 
            '-J', self.active_slurm_job_name, 
            '--parsable', '-o', slurm_output_filepath]
            container_launch_params = [slurm_script_filepath,
                                       "--team-name", "{}".format(actor.name),
                                       "--team-email", "{}".format(actor.email),
                                       "--submission-filepath", "{}".format(submission_filepath),
                                       "--result-dirpath", self.execution_results_dirpath,
                                       "--trojai-config-filepath", trojai_config_filepath,
                                       "--leaderboard-name", self.leaderboard_name,
                                       "--data-split-name", self.data_split_name,
                                       "--trojai-test-harness-dirpath", test_harness_dirpath,
                                       "--python-exec", python_executable,
                                       "--task-executor-filepath", task_executor_script_filepath,
                                       "--is-local",
                                       "--custom-home", "{}".format(custom_home_dirpath),
                                       "--custom-scratch", "{}".format(custom_scratch_dirpath)]
        else:
            sbatch_control_params = ['sbatch', 
            '--partition', control_slurm_queue, 
            '--parsable', 
            '--nice={}'.format(self.slurm_nice), 
            '--nodes', '1', 
            '--ntasks-per-node', '1', 
            '--cpus-per-task', '1', ':']
            sbatch_vm_params = ['--partition', self.slurm_queue_name, 
            '--nice={}'.format(self.slurm_nice), 
            '--nodes', '1', 
            '--ntasks-per-node', '1', 
            '--cpus-per-task', str(cpus_per_task), 
            '--exclusive', '-J', self.active_slurm_job_name, 
            '--parsable', '-o', slurm_output_filepath]
            container_launch_params = [slurm_script_filepath,
                                       "--team-name", "{}".format(actor.name),
                                       "--team-email", "{}".format(actor.email),
                                       "--submission-filepath", "{}".format(submission_filepath),
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

        rc = subprocess.run(cmd_str_list, capture_output=True)
        stdout = rc.stdout
        stderr = rc.stderr
        # out = subprocess.Popen(cmd_str_list,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE)

        # stdout, stderr = out.communicate()

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

    def get_result_table_row(self, results_manager: ResultsManager, a: Airium, actor: Actor, leaderboard: Leaderboard, g_drive: DriveIO):
        if self.active_slurm_job_name is not None:
            return

        submission_timestr = time_utils.convert_epoch_to_iso(self.submission_epoch)

        df = leaderboard.load_results_df(results_manager)
        filtered_df = results_manager.filter_primary_key(df, self.get_submission_epoch_str_primary(), self.data_split_name, actor.uuid)

        if filtered_df is None:
            logging.error('Unable to find submission: {} {} {} for actor {} on leaderboad {}'.format(self.get_submission_epoch_str_primary(), self.data_split_name, actor.uuid, actor.name, leaderboard.name))
            return

        if len(filtered_df) > 1:
            logging.error('Found multiple entries that match {} {} {} for actor {} on leaderboard {}'.format(self.get_submission_epoch_str_primary(), self.data_split_name, actor.uuid, actor.name, leaderboard.name))
            return

        if self.g_file.modified_epoch == 0 or self.g_file.modified_epoch is None:
            file_timestr = "None"
        else:
            file_timestr = time_utils.convert_epoch_to_iso(self.g_file.modified_epoch)

        if len(self.web_display_execution_errors.strip()) == 0:
            self.web_display_execution_errors = "None"

        if len(self.web_display_parse_errors.strip()) == 0:
            self.web_display_parse_errors = "None"

        with a.tr():
            a.td(_t=actor.name)

            submission_metrics = leaderboard.submission_metrics
            for metric_name, metric in submission_metrics.items():
                if metric.write_html:
                    metric_value = filtered_df[metric_name].values[0]

                    if metric_value is None or math.isnan(float(metric_value)):
                        metric_value = ''

                    if isinstance(metric_value, float):
                        a.td(_t=str(round(metric_value, metric.html_decimal_places)))
                    else:
                        a.td(_t=str(metric_value))

            rounded_execution_time = self.execution_runtime
            if rounded_execution_time is not None:
                rounded_execution_time = round(rounded_execution_time, 2)

            if rounded_execution_time is None or math.isnan(float(rounded_execution_time)):
                rounded_execution_time = None

            a.td(_t=rounded_execution_time)
            a.td(_t=submission_timestr)
            a.td(_t=file_timestr)

            if not hasattr(self, 'leaderboard_revision'):
                self.leaderboard_revision = leaderboard.revision

            a.td(_t='Rev{}'.format(self.leaderboard_revision))
            a.td(_t=self.web_display_parse_errors)
            a.td(_t=self.web_display_execution_errors)

    def has_new_metrics(self, leaderboard: Leaderboard) -> bool:
        submission_metrics = leaderboard.submission_metrics
        for metric_name, metric in submission_metrics.items():
            if metric_name not in self.processed_metric_names:
                return True

        return False

    def compute_missing_metrics(self, results_manager: ResultsManager, actor: Actor, leaderboard: Leaderboard, g_drive: DriveIO):
        if self.has_new_metrics(leaderboard):
            errors, new_processed_metrics = leaderboard.process_metrics(g_drive, results_manager, self.data_split_name, self.execution_results_dirpath, actor.name, actor.uuid, self.get_submission_epoch_str_primary(), self.processed_metric_names)
            self.processed_metric_names.extend(new_processed_metrics)

        # TODO: Should we update the errors for the submission (will have to be careful to not repeat errors)


class SubmissionManager(object):
    def __init__(self, leaderboard_name):
        self.leaderboard_name: str = leaderboard_name
        # keyed on uuid
        self.__submissions: Dict[str, List[Submission]] = dict()

    def __str__(self):
        msg = ""
        for a, submissions in self.__submissions.items():
            msg = msg + "Actor: {}: \n".format(a)
            for s in submissions:
                msg = msg + "  " + s.__str__() + "\n"
        return msg

    def check_for_new_metrics(self, results_manager: ResultsManager, leaderboard: Leaderboard, actor_manager: ActorManager, g_drive: DriveIO):
        for actor_uuid, submissions in self.__submissions.items():
            actor = actor_manager.get_from_uuid(actor_uuid)

            for submission in submissions:
                if submission.active_slurm_job_name is not None:
                    continue
                submission.compute_missing_metrics(results_manager, actor, leaderboard, g_drive)

    def gather_submissions(self, results_manager: ResultsManager, leaderboard: Leaderboard, data_split_name:str, metric_name: str, metric_criteria: float, actor: Actor, g_drive: DriveIO) -> List[Submission]:
        gathered_submissions = list()

        actor_submissions = self.__submissions[actor.uuid]
        submission_metrics = leaderboard.submission_metrics
        metric = submission_metrics[metric_name]

        result_df = leaderboard.load_results_df(results_manager)

        for submission in actor_submissions:
            if submission.data_split_name == data_split_name:
                filtered_result_df = results_manager.filter_primary_key(result_df, submission.get_submission_epoch_str_primary(), data_split_name, actor.uuid)
                if metric.store_result:
                    if filtered_result_df[metric.get_name()] is not None:
                        metric_value = filtered_result_df[metric.get_name()].values[0]

                        if metric.compare(metric_value, metric_criteria):
                            gathered_submissions.append(submission)

        return gathered_submissions

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

    def has_submission_file_id(self, actor: Actor, new_modified_epoch):
        submissions = self.get_submissions_by_actor(actor)

        for submission in submissions:
            if submission.g_file.modified_epoch == new_modified_epoch:
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

    def write_score_table_unique(self, output_dirpath, results_manager: ResultsManager, leaderboard: Leaderboard, actor_manager: ActorManager, data_split_name: str, g_drive: DriveIO):

        result_filename = 'results-unique-{}-{}.html'.format(leaderboard.name, data_split_name)
        result_filepath = os.path.join(output_dirpath, leaderboard.name, result_filename)

        a = Airium()

        valid_submissions : Dict[str, List[Submission]] = {}

        for actor_uuid, submission_list in self.__submissions.items():
            valid_submissions[actor_uuid] = list()

            for submission in submission_list:
                if submission.data_split_name == data_split_name:
                    valid_submissions[actor_uuid].append(submission)

        evaluation_metric_name = leaderboard.evaluation_metric_name
        submission_metrics = leaderboard.submission_metrics

        df = leaderboard.load_results_df(results_manager)

        with a.div(klass='card-body card-body-cascade pb-0'):
            a.h2(klass='pb-q card-title', _t='Best Results based on {}'.format(evaluation_metric_name))
            with a.div(klass='table-responsive'):
                with a.table(id='{}-{}-results'.format(leaderboard.name, data_split_name), klass='table table-striped table-bordered table-sm'):
                    with a.thead():
                        with a.tr():
                            a.th(klass='th-sm', _t='Team')
                            for metric_name, metric in submission_metrics.items():
                                if metric.write_html:
                                    a.th(klass='th-sm', _t=metric_name)

                            a.th(klass='th-sm', _t='Runtime (s)')
                            a.th(klass='th-sm', _t='Submission Timestamp')
                            a.th(klass='th-sm', _t='File Timestamp')
                            a.th(klass='th-sm', _t='Leaderboard Revision')
                            a.th(klass='th-sm', _t='Parsing Errors')
                            a.th(klass='th-sm', _t='Launch Errors')
                    with a.tbody():
                        metric = submission_metrics[evaluation_metric_name]

                        for actor_uuid, submissions in valid_submissions.items():
                            best_submission_score = None
                            best_submission = None
                            actor = actor_manager.get_from_uuid(actor_uuid)
                            for s in submissions:
                                if s.active_slurm_job_name is not None:
                                    continue

                                # TODO: We could probably optimize this
                                filtered_df = results_manager.filter_primary_key(df, s.get_submission_epoch_str_primary(), data_split_name, actor_uuid)
                                if filtered_df is None:
                                    print('Why you none...')
                                if filtered_df[evaluation_metric_name] is not None:
                                    value = filtered_df[evaluation_metric_name].values[0]
                                    if best_submission_score is None or metric.compare(value, best_submission_score):
                                        best_submission_score = value
                                        best_submission = s

                            if best_submission is not None:
                                best_submission.get_result_table_row(results_manager, a, actor, leaderboard, g_drive)

        with open(result_filepath, 'w') as f:
            f.write(str(a))

        return result_filepath

    def write_score_table(self, output_dirpath, result_manager: ResultsManager, leaderboard: Leaderboard, actor_manager: ActorManager, data_split_name: str, g_drive: DriveIO):
        result_filename = 'results-{}-{}.html'.format(leaderboard.name, data_split_name)
        result_filepath = os.path.join(output_dirpath, leaderboard.name, result_filename)
        a = Airium()

        valid_submissions: Dict[str, List[Submission]] = {}
        submission_metrics = leaderboard.submission_metrics

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
                            for metric_name, metric in submission_metrics.items():
                                if metric.write_html:
                                    a.th(klass='th-sm', _t=metric_name)

                            a.th(klass='th-sm', _t='Runtime (s)')
                            a.th(klass='th-sm', _t='Submission Timestamp')
                            a.th(klass='th-sm', _t='File Timestamp')
                            a.th(klass='th-sm', _t='Leaderboard Revision')
                            a.th(klass='th-sm', _t='Parsing Errors')
                            a.th(klass='th-sm', _t='Launch Errors')
                    with a.tbody():
                        for actor_uuid, submissions in valid_submissions.items():
                            actor = actor_manager.get_from_uuid(actor_uuid)
                            for s in submissions:
                                s.get_result_table_row(result_manager, a, actor, leaderboard, g_drive)

        with open(result_filepath, 'w') as f:
            f.write(str(a))

        return result_filepath


    def generate_round_results_csv(self, results_manager: ResultsManager, leaderboard: Leaderboard, actor_manager: ActorManager, overwrite_csv: bool = False):
        if os.path.exists(leaderboard.summary_results_csv_filepath) and not overwrite_csv:
            result_df = leaderboard.load_summary_results_csv_into_df()
        else:
            result_df = None

        num_dfs_added = 0

        new_data = dict()

        dictionary_time_start = time.time()

        for actor in actor_manager.get_actors():
            submissions = self.get_submissions_by_actor(actor)
            for data_split in leaderboard.get_all_data_split_names():
                for submission in submissions:
                    if submission.active_slurm_job_name is not None:
                        continue

                    if submission.data_split_name == data_split:

                        temp_new_data = leaderboard.update_results_csv(result_df, results_manager, submission.submission_epoch, data_split, actor.name, actor.uuid)

                        if len(temp_new_data) > 0:
                            num_dfs_added += 1
                            for key in temp_new_data.keys():
                                if key in new_data:
                                    if isinstance(temp_new_data[key], list):
                                        new_data[key].extend(temp_new_data[key])
                                    elif isinstance(temp_new_data[key], np.ndarray):
                                        new_data[key] = np.concatenate((new_data[key], temp_new_data[key]))
                                else:
                                    new_data[key] = temp_new_data[key]


        dictionary_time_end = time.time()

        logging.info('Total submissions added = {}, time to create dictionary: {}'.format(num_dfs_added, dictionary_time_end-dictionary_time_start))

        df_time_start = time.time()

        if num_dfs_added > 0:
            if result_df is None:
                result_df = pd.DataFrame(new_data)
            else:
                new_result_df = pd.DataFrame(new_data)
                result_df = pd.concat([result_df, new_result_df], ignore_index=True)

            result_df.to_csv(leaderboard.summary_results_csv_filepath, index=False)

        # If there have been no submissions, then we should just create an empty one
        if not os.path.exists(leaderboard.summary_results_csv_filepath):
            with open(leaderboard.summary_results_csv_filepath, 'w') as fp:
                pass

        df_time_end = time.time()

        logging.info('Finished processing round results for {}, time to create df: {}'.format(leaderboard.summary_results_csv_filepath, df_time_end-df_time_start))

        return result_df

    def recompute_metrics(self, trojai_config: TrojaiConfig, results_manager: ResultsManager, leaderboard: Leaderboard):

        actor_manager = ActorManager.load_json(trojai_config)
        g_drive = DriveIO(trojai_config.token_pickle_filepath)

        for actor_uuid, submissions in self.__submissions.items():
            actor = actor_manager.get_from_uuid(actor_uuid)
            for submission in submissions:
                # Verify it is not active prior to computing metrics
                if submission.active_slurm_job_name is None:
                    # This should recompute all metrics
                    errors, new_processed_metrics = leaderboard.process_metrics(g_drive, results_manager, submission.data_split_name, submission.execution_results_dirpath, actor.name, actor.uuid, submission.get_submission_epoch_str_primary(), processed_metrics=[])
                    submission.processed_metric_names = new_processed_metrics

        results_manager.save_all()
        self.save_json(leaderboard)

    def dump_metaparameter_csv(self, trojai_config: TrojaiConfig, leaderboard: Leaderboard):
        actor_manager = ActorManager.load_json(trojai_config)

        for actor_uuid, submissions in self.__submissions.items():
            actor = actor_manager.get_from_uuid(actor_uuid)
            for submission in submissions:
                submission.dump_summary_schema_csv(trojai_config, actor.name, leaderboard)



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
    results_manager = ResultsManager()

    leaderboard_names = []

    if args.name is not None:
        leaderboard_names.append(args.name)
    else:
        for name in trojai_config.archive_leaderboard_names:
            leaderboard_names.append(name)
        for name in trojai_config.active_leaderboard_names:
            leaderboard_names.append(name)


    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

    print('Attempting to acquire PID file lock.')
    lock_file = '/var/lock/trojai-lockfile'
    if args.unsafe:
        for name in leaderboard_names:
            leaderboard = Leaderboard.load_json(trojai_config, name)
            submission_manager = SubmissionManager.load_json(leaderboard)
            submission_manager.recompute_metrics(trojai_config, results_manager, leaderboard)
            print('Finished recomputing metrics for {}'.format(leaderboard.name))
    else:
        with open(lock_file, 'w') as f:
            try:
                fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print('  PID lock acquired')
                for name in leaderboard_names:
                    leaderboard = Leaderboard.load_json(trojai_config, name)
                    submission_manager = SubmissionManager.load_json(leaderboard)
                    submission_manager.recompute_metrics(trojai_config, results_manager, leaderboard)
                    print('Finished recomputing metrics for {}'.format(leaderboard.name))
            except OSError as e:
                print('check-and-launch was already running when called. {}'.format(e))
            finally:
                fcntl.lockf(f, fcntl.LOCK_UN)

def dump_metaparameters_csv(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

    print('Attempting to acquire PID file lock.')
    lock_file = '/var/lock/trojai-lockfile'

    with open(lock_file, 'w') as f:
        try:
            fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print('  PID lock acquired')

            leaderboard_name = args.name
            leaderboards = []

            if leaderboard_name is None:
                for leaderboard_name in trojai_config.active_leaderboard_names:
                    leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
                    leaderboards.append(leaderboard)

                for leaderboard_name in trojai_config.archive_leaderboard_names:
                    leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
                    leaderboards.append(leaderboard)

            else:
                leaderboard = Leaderboard.load_json(trojai_config, args.name)
                leaderboards.append(leaderboard)

            for leaderboard in leaderboards:
                submission_manager = SubmissionManager.load_json(leaderboard)
                submission_manager.dump_metaparameter_csv(trojai_config, leaderboard)
                print('Finished dumping metaparameters csv for {}'.format(leaderboard.name))
        except OSError as e:
            print('check-and-launch was already running when called. {}'.format(e))
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)

def generate_results_csv(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    actor_manager = ActorManager.load_json(trojai_config)
    leaderboard = Leaderboard.load_json(trojai_config, args.name)
    submission_manager = SubmissionManager.load_json(leaderboard)
    results_manager = ResultsManager()

    submission_manager.generate_round_results_csv(results_manager, leaderboard, actor_manager, overwrite_csv=False)

def update_configuration_latest(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    actor_manager = ActorManager.load_json(trojai_config)

    leaderboard_names = []

    if args.name is not None:
        leaderboard_names.append(args.name)
    else:
        for name in trojai_config.archive_leaderboard_names:
            leaderboard_names.append(name)
        for name in trojai_config.active_leaderboard_names:
            leaderboard_names.append(name)

    for name in leaderboard_names:
        leaderboard = Leaderboard.load_json(trojai_config, name)

        old_submission_config = None
        backup_filepath = os.path.join(os.path.dirname(leaderboard.submissions_filepath), '{}_submissions_backup.json'.format(leaderboard.name))

        if not os.path.exists(leaderboard.submissions_filepath):
            print('Warning: Unable to find {}, skipping...'.format(leaderboard.submissions_filepath))
            continue

        with open(leaderboard.submissions_filepath, 'r') as fp:
            old_submission_config = json.load(fp)

        if os.path.exists(backup_filepath):
            print('Error, backup already exists, cancelling')
            # return

        with open(backup_filepath, 'w') as fp:
            json.dump(old_submission_config, fp, indent=2)


        new_submission_manager = SubmissionManager(leaderboard.name)

        if '_SubmissionManager__submissions' in old_submission_config:
            submissions_dict = old_submission_config['_SubmissionManager__submissions']
            for actor_uuid, submissions_list in submissions_dict.items():
                for submission_dict in submissions_list:
                    actor = actor_manager.get_from_uuid(actor_uuid)

                    g_email = None
                    g_filename = None
                    g_file_id = None
                    now = datetime.datetime.now()
                    g_temp_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
                    g_timestamp = None

                    if 'g_file' in submission_dict:
                        g_file_dict = submission_dict['g_file']
                        g_email = get_value(g_file_dict, 'email')
                        g_filename = get_value(g_file_dict, 'name')
                        g_file_id = get_value(g_file_dict, 'id')
                        g_timestamp = get_value(g_file_dict, 'modified_epoch')

                    if g_email is None or g_filename is None or g_file_id is None or g_timestamp is None:
                        print('Failed to parse g_file for {}... skipping'.format(leaderboard.submissions_filepath))
                        continue

                    g_file = GoogleDriveFile(g_email, g_filename, g_file_id, g_temp_timestamp)
                    g_file.modified_epoch = g_timestamp

                    data_split_name = get_value(submission_dict, 'data_split_name')
                    provenance = get_value(submission_dict, 'provenance')
                    submission_epoch = get_value(submission_dict, 'submission_epoch')
                    slurm_queue_name = get_value(submission_dict, 'slurm_queue_name')

                    if data_split_name is None or provenance is None or submission_epoch is None or slurm_queue_name is None:
                        print('Failed to parse parameters for {}... skipping'.format(leaderboard.submissions_filepath))
                        continue

                    new_submission = Submission(g_file, actor, leaderboard, data_split_name, provenance, submission_epoch, slurm_queue_name)

                    # Remove unneeded attributes
                    del submission_dict['g_file']
                    del submission_dict['py/object']

                    # Copy over all attributes
                    update_object_values(new_submission, submission_dict)

                    new_submission_manager.add_submission(actor, new_submission)

        new_submission_manager.save_json(leaderboard)
        print('finished writing updated submission manager: {}'.format(leaderboard.name))


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
    recompute_metrics_parser.add_argument('--name', type=str, help='The name of the leaderboards or None if rerun all leaderboards', required=False, default=None)
    recompute_metrics_parser.add_argument('--unsafe', action='store_true', help='Disables trojai lock (useful for debugging only)')
    recompute_metrics_parser.set_defaults(func=recompute_metrics)

    generate_results_csv_parser = subparser.add_parser('generate-results-csv', help='Generates the RESULTS CSV for a round')
    generate_results_csv_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    generate_results_csv_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    generate_results_csv_parser.set_defaults(func=generate_results_csv)

    dump_metaparameters_csv_parser = subparser.add_parser('dump-metaparameters')
    dump_metaparameters_csv_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    dump_metaparameters_csv_parser.add_argument('--name', type=str, help='The name of the leaderboards', default=None)
    dump_metaparameters_csv_parser.set_defaults(func=dump_metaparameters_csv)

    update_config_parser = subparser.add_parser('update-config')
    update_config_parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the main trojai config', required=True)
    update_config_parser.add_argument('--name', type=str, help='The name of the leaderboard to update, if not specified then will update all leaderboards', required=False, default=None)
    update_config_parser.set_defaults(func=update_configuration_latest)

    args = parser.parse_args()
    args.func(args)
