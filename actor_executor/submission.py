# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import subprocess
import logging
from typing import List
from typing import Dict

from actor_executor.drive_io import DriveIO
from actor_executor.google_drive_file import GoogleDriveFile
from actor_executor.actor import Actor
from actor_executor import json_io
from actor_executor import slurm
from actor_executor import time_utils
from actor_executor import ground_truth


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

    def check_submission(self, g_drive: DriveIO, log_file_byte_limit: int) -> None:

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
            ground_truth.process_results(self, g_drive, log_file_byte_limit)

            if self.slurm_queue == 'sts':
                # delete the container file to avoid filling up disk space for the STS server
                time_str = time_utils.convert_epoch_to_psudo_iso(self.execution_epoch)
                submission_filepath = os.path.join(self.global_submission_dirpath, self.actor.name, time_str, self.file.name)
                logging.info('Deleting container image: "{}"'.format(submission_filepath))
                os.remove(submission_filepath)
        else:
            logging.warning("Incorrect format for stdout from squeue: {}".format(stdoutSplitNL))

            # attempt to process the result
            ground_truth.process_results(self, g_drive, log_file_byte_limit)

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
                    scores.append([s.actor.name, s.cross_entropy, s.cross_entropy_95_confidence_interval, s.brier_score, s.roc_auc, s.execution_runtime, execute_timestr, file_timestr, s.web_display_parse_errors, s.web_display_execution_errors])
        return scores
