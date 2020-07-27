import traceback
import logging
import logging.handlers
import os

import fcntl
from config import Config

from drive_io import DriveIO
from actor import Actor, ActorManager
from submission import Submission, SubmissionManager
import html_output
import time_utils


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
      actor.job_status = "None"

    logging.info("Checking for new submissions from {}.".format(actor.name))

    # query drive for a submission by this actor
    actor_file_list = g_drive.query_by_email(actor.email)

    # filter list based on file prefix
    sts = config.slurm_queue == 'sts'

    gdrive_file_list = list()
    for g_file in actor_file_list:
        if sts and g_file.name.startswith('test'):
            gdrive_file_list.append(g_file)
        if not sts and not g_file.name.startswith('test'):
            gdrive_file_list.append(g_file)

    # ensure submission is unique (one and only one possible submission file from a team email)
    if len(gdrive_file_list) < 1:
        logging.info("Actor {} does not have a submission from email {}.".format(actor.name, actor.email))
        actor.file_status = "None"

    if len(gdrive_file_list) > 1:
        logging.warning("Actor {} shared {} files from email {}.".format(actor.name, len(gdrive_file_list),actor.email))
        actor.file_status = "Multiple files shared"

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
                logging.info('Submission is different .... EXECUTING; new file name: {}, new file epoch: {}, last file epoch: {}'.format(g_file.name, g_file.modified_epoch, actor.last_file_epoch))
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
            submission.check_submission(g_drive, config.log_file_byte_limit)

    # look for any new submissions
    # This might modify the SubmissionManager instance
    logging.info('Done processing old/pending submission.')
    process_new_submission(config, g_drive, actor, submission_manager, config_filepath, cur_epoch)


def main(config: Config, push_html: bool, config_filepath: str) -> None:
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

    logging.debug('Updating website.')
    html_output.update_html(config.html_repo_dir, actor_manager, submission_manager, config.execute_window, config.job_table_name, config.result_table_name, push_html, cur_epoch, config.accepting_submissions, config.slurm_queue)

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

    parser.add_argument("--no-push-html", dest='push_html',
                        help="Disables pushing the html web content to the Internet",
                        action='store_false')

    parser.set_defaults(push_html=True)

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
                                format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                                handlers=[handler])

            logging.debug('PID file lock acquired in directory {}'.format(config.submission_dir))
            main(config, args.push_html, args.config_file)
        except OSError as e:
            print('Server "{}", check-and-launch was already running when called.'.format(config.slurm_queue))
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


