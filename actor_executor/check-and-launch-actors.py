# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import traceback
import logging
import logging.handlers
import os

import fcntl
from actor_executor.config import Config

from actor_executor.drive_io import DriveIO
from actor_executor.actor import Actor, ActorManager
from actor_executor.submission import Submission, SubmissionManager
from actor_executor import html_output
from actor_executor import time_utils


def process_new_submission(config: Config, g_drive: DriveIO, actor: Actor, submission_manager: SubmissionManager, config_filepath: str, cur_epoch: int) -> None:
    if not config.accepting_submissions:
        logging.info("New submissions are closed.")
        return

    if actor.in_queue(config.slurm_queue):
        logging.info("Job is currently in queue for {}.".format(actor.name))
        return

    if actor.is_disabled():
        logging.info("{} is currently marked as disabled".format(actor.name))
        actor.job_status = "Disabled"
        return
    elif actor.job_status == "Disabled":
        logging.info("{} was previously disabled, resetting job status back to None.".format(actor.name))
        actor.job_status = "None"

    logging.info("Checking for new submissions from {}.".format(actor.name))

    # query drive for a submission by this actor
    actor_file_list = g_drive.query_by_email(actor.email)

    # filter list based on file prefix
    sts_flag = config.slurm_queue == 'sts'

    gdrive_file_list = list()
    for g_file in actor_file_list:
        if sts_flag and g_file.name.startswith('test'):
            gdrive_file_list.append(g_file)
        if not sts_flag and not g_file.name.startswith('test'):
            gdrive_file_list.append(g_file)

    # ensure submission is unique (one and only one possible submission file from a team email)
    if len(gdrive_file_list) < 1:
        logging.info("Actor {} does not have a submission from email {}.".format(actor.name, actor.email))
        actor.file_status = "None"
        actor.job_status = "None"

    if len(gdrive_file_list) > 1:
        logging.warning("Actor {} shared {} files from email {}.".format(actor.name, len(gdrive_file_list),actor.email))
        actor.file_status = "Multiple files shared"
        actor.job_status = "None"

    if len(gdrive_file_list) == 1:
        g_file = gdrive_file_list[0]
        logging.info('Detected submission from actor {}: {}'.format(actor.email, g_file.name))
        actor.file_status = "Ok"

        # Check file timestamp at 1 second resolution
        if not actor.can_submit_timewindow(config.execute_window, cur_epoch) and int(g_file.modified_epoch) != int(actor.last_file_epoch):
            logging.info('Team {} timeout window has not elapsed. cur_epoch: {}, last_exec_epoc: {}'.format(actor.name, cur_epoch, actor.last_execution_epoch))
            actor.job_status = "Awaiting Timeout"
        else:
            if int(g_file.modified_epoch) != int(actor.last_file_epoch):
                logging.info('Submission timestamp is different .... EXECUTING; new file name: {}, new file epoch: {}, last file epoch: {}'.format(g_file.name, g_file.modified_epoch, actor.last_file_epoch))
                submission = Submission(g_file, actor, config.submission_dir, config.results_dir, config.ground_truth_dir, config.slurm_queue)
                submission_manager.add_submission(submission)
                logging.info('Added submission file name "{}" to manager from email "{}"'.format(submission.file.name, actor.email))
                submission.execute(config.slurm_script_file, config_filepath, cur_epoch)
            else:
                logging.info('Submission found is the same as the last execution run for team {}; new file name: {}, new file epoch: {}, last file epoch: {}'.format(actor.name, g_file.name, g_file.modified_epoch, actor.last_file_epoch))
                actor.job_status = "None"


def process_team(config: Config, g_drive: DriveIO, actor: Actor, submission_manager: SubmissionManager, config_filepath: str, cur_epoch: int) -> None:

    actor_submission_list = submission_manager.get_submissions_by_actor(actor)
    logging.debug('Found {} old submissions for actor {}.'.format(len(actor_submission_list), actor.name))
    for submission in actor_submission_list:
        # if the actor has any in flight submissions
        if submission.slurm_job_name is not None:
            # re link actor object which was lost on loading object from json serialization
            submission.actor = actor
            logging.info('Found live submission "{}" from "{}"'.format(submission.file.name, submission.actor.name))
            submission.check(g_drive, config.log_file_byte_limit)

    # look for any new submissions
    # This might modify the SubmissionManager instance
    logging.info('Done processing old/pending submission.')
    process_new_submission(config, g_drive, actor, submission_manager, config_filepath, cur_epoch)


def main(config: Config, config_filepath: str) -> None:
    cur_epoch = time_utils.get_current_epoch()

    if config.slurm_queue == 'sts':
        logging.info('STS -  Check and Launch Actors')
    else:
        logging.info('ES  - Check and Launch Actors')
    logging.debug('Slurm Queue = "{}"'.format(config.slurm_queue))

    # load the instance of ActorManager from the serialized json file
    actor_manager = ActorManager.load_json(config.actor_json_file)
    logging.debug('Loaded actor_manager from filepath: {}'.format(config.actor_json_file))
    logging.debug(actor_manager)

    # load the instance of SubmissionManager from the serialized json file
    submission_manager = SubmissionManager.load_json(config.submissions_json_file)
    logging.debug('Loaded submission_manager from filepath: {}'.format(config.submissions_json_file))
    logging.debug(submission_manager)

    logging.info('Actor Manger has {} actors.'.format(len(actor_manager.get_keys())))
    logging.info('Submissions Manger has {} actors and {} total submissions.'.format(submission_manager.get_number_actors(), submission_manager.get_number_submissions()))

    g_drive = DriveIO(config.token_pickle_file)
    # Loop over actors, checking if there is a submission for each
    for key in actor_manager.get_keys():
        try:
            actor = actor_manager.get(key)
            logging.info('**************************************************')
            logging.info('Processing {}:'.format(actor.name))
            logging.info('**************************************************')
            process_team(config, g_drive, actor, submission_manager, config_filepath, cur_epoch)
        except:
            msg = 'Exception processing actor "{}" loop:\n{}'.format(key, traceback.format_exc())
            logging.error(msg)

    # Write all updates to actors back to file
    logging.debug('Serializing updated actor_manger back to json.')
    actor_manager.save_json(config.actor_json_file)

    logging.debug('Serializing updated submission_manager back to json.')
    submission_manager.save_json(config.submissions_json_file)

    if config.slurm_queue == 'sts':
        logging.info('STS -  Finished Check and Launch Actors')
    else:
        logging.info('ES  - Finished Check and Launch Actors')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check and Launch script for TrojAI challenge participants')
    parser.add_argument('--config-file', type=str,
                        help='The JSON file that describes all actors',
                        default='config.json')

    args = parser.parse_args()

    config = Config.load_json(args.config_file)

    # PidFile ensures that this script is only running once
    print('Attempting to acquire PID file lock.')
    lock_file = '/var/lock/trojai-{}-lockfile'.format(config.slurm_queue)
    with open(lock_file, 'w') as f:
        try:
            fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print('  PID lock acquired')
            # make sure intermediate folders to the logfile exists
            parent_fp = os.path.dirname(config.log_file)
            if not os.path.exists(parent_fp):
                os.makedirs(parent_fp)
            # Add the log message handler to the logger
            handler = logging.handlers.RotatingFileHandler(config.log_file, maxBytes=100*1e6, backupCount=10) # 100MB
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                                handlers=[handler])

            logging.debug('PID file lock acquired in directory {}'.format(config.submission_dir))
            main(config, args.config_file)
        except OSError as e:
            print('Server "{}", check-and-launch was already running when called.'.format(config.slurm_queue))
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


