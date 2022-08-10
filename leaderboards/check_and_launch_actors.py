# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import time
import traceback
import logging
import logging.handlers
import os

from typing import Dict

import fcntl

from leaderboards.trojai_config import TrojaiConfig
from leaderboards.drive_io import DriveIO
from leaderboards.actor import Actor, ActorManager
from leaderboards.submission import Submission, SubmissionManager
from leaderboards import time_utils
from leaderboards.leaderboard import Leaderboard
from leaderboards.html_output import update_html_pages


def process_new_submission(trojai_config: TrojaiConfig, g_drive: DriveIO, actor: Actor, active_leaderboards: Dict[str, Leaderboard],  active_submission_managers: Dict[str, SubmissionManager]) -> None:

    if not trojai_config.accepting_submissions:
        logging.info("New submissions are closed.")
        return

    if actor.is_disabled():
        logging.info("{} is currently marked as disabled".format(actor.name))
        actor.update_all_job_status('Disabled')
        return
    elif actor.has_job_status('Disabled'):
        logging.info("{} was previously disabled, resetting job status back to None.".format(actor.name))
        actor.update_all_job_status('None', check_value='Disabled')

    logging.info("Checking for new submissions from {}.".format(actor.name))

    # query drive for a submission by this actor
    actor_file_list = g_drive.query_by_email(actor.email)

    # Search for entries that contain a valid leaderboards name and dataset split
    has_general_errors = False
    valid_submissions = {}

    # Setup all valid submissions
    for leaderboard_name, leaderboard in active_leaderboards.items():
        for data_split_name in leaderboard.get_submission_data_split_names():
            key = '{}_{}'.format(leaderboard_name, data_split_name)
            valid_submissions[key] = []

    # Find valid files for submission
    # TODO: Can we improve the general file status error? Currently it captures multiple error scenarios: filename split valid, correct leaderboards, if an actor can submit
    for g_file in actor_file_list:
        filename = g_file.name
        filename_split = filename.split('_')

        # Expected format is leaderboards-name_data-split-name_container-name.simg
        if len(filename_split) <= 2:
            logging.info('File {} from actor {} did not have expected format'.format(filename, actor.name))
            actor.general_file_status = 'Shared File Error'
            has_general_errors = True
            continue

        leaderboard_name = filename_split[0]
        data_split_name = filename_split[1]

        # check if valid leaderboards
        if leaderboard_name not in active_leaderboards.keys():
            logging.info('File {} from actor {} did not have a valid leaderboards name: {}'.format(filename, actor.name, leaderboard_name))
            if not has_general_errors:
                actor.general_file_status = 'Shared File Error'
                has_general_errors = True
            continue

        leaderboard = active_leaderboards[leaderboard_name]
        submission_manager = active_submission_managers[leaderboard_name]

        # Check if valid data split name
        if not leaderboard.can_submit_to_dataset(data_split_name):
            logging.info('File {} from actor {} did not have a valid data split name: {}'.format(filename, actor.name, data_split_name))
            if not has_general_errors:
                actor.general_file_status = 'Shared File Error'
                has_general_errors = True
            continue

        # Check if actor already has a job waiting to be processed (may be in queue)
        if submission_manager.has_active_submission(actor):
            logging.info('Detected another submission for {}, named {}, but an active submission is in progress'.format(actor.name, filename))
            continue

        key = '{}_{}'.format(leaderboard_name, data_split_name)
        if key not in valid_submissions.keys():
            logging.info('Unknown leaderboards key when adding valid submissions: {}'.format(key))
            continue

        valid_submissions[key].append(g_file)

    if not has_general_errors:
        actor.general_file_status = 'Ok'

    # Check timestamps and multiple entries
    for key, g_file_list in valid_submissions.items():

        key_split = key.split('_')
        leaderboard_name = key_split[0]
        data_split_name = key_split[1]

        leaderboard = active_leaderboards[leaderboard_name]
        submission_manager = active_submission_managers[leaderboard_name]

        if len(g_file_list) == 0:
            logging.info('Actor {} does not have a submission from email {} for leaderboards {} and data split {}.'.format(actor.name, actor.email, leaderboard_name, data_split_name))
            actor.update_file_status(leaderboard_name, data_split_name, 'None')
            actor.update_job_status(leaderboard_name, data_split_name, 'None')

        if len(g_file_list) > 1:
            logging.warning('Actor {} shared {} files from email {} for leaderboards {} and data split {}'.format(actor.name, len(g_file_list), actor.email, leaderboard_name, data_split_name))
            actor.update_file_status(leaderboard_name, data_split_name, 'Multiple files shared')
            actor.update_job_status(leaderboard_name, data_split_name, 'None')

        if len(g_file_list) == 1:
            g_file = g_file_list[0]
            logging.info('Detected submission from actor {}: {} for leaderboards {} and data split {}'.format(actor.name, g_file.name, leaderboard_name, data_split_name))
            actor.update_file_status(leaderboard_name, data_split_name, 'Ok')

            # Check timestamp (1 second granularity)
            exec_epoch = time_utils.get_current_epoch()

            # Sleep for 1 second to have distinct execu epochs
            time.sleep(1)

            if not actor.can_submit_time_window(leaderboard_name, data_split_name, leaderboard.get_timeout_window_time(data_split_name), exec_epoch) and int(g_file.modified_epoch) != int(actor.get_last_file_epoch(leaderboard_name, data_split_name)):
                logging.info('Team {} timeout window has not elapsed. exec_epoch: {}, last_exec_epoch: {}, leaderboards: {}, data split: {}'.format(actor.name, exec_epoch, actor.get_last_execution_epoch(leaderboard_name, data_split_name), leaderboard_name, data_split_name))
                actor.update_job_status(leaderboard_name, data_split_name, 'Awaiting Timeout')
            else:
                if int(g_file.modified_epoch) != int(actor.get_last_file_epoch(leaderboard_name, data_split_name)):
                    logging.info('Submission timestamp is different .... EXECUTING; new file name: {}, new file epoch: {}, last file epoch: {}'.format(g_file.name, g_file.modified_epoch, actor.get_last_file_epoch(leaderboard_name, data_split_name)))
                    submission = Submission(g_file, actor, leaderboard, data_split_name)
                    submission_manager.add_submission(actor, submission)
                    logging.info('Added submission file name "{}" to manager from email "{}"'.format(submission.g_file.name, actor.email))
                    submission.execute(actor, trojai_config, exec_epoch)
                else:
                    logging.info('Submission found is the same as the last execution run for team {}; new file name: {}, new file epoch: {}, last file epoch: {}'.format(actor.name, g_file.name, g_file.modified_epoch, actor.get_last_file_epoch(leaderboard_name, data_split_name)))
                    actor.update_job_status(leaderboard_name, data_split_name, 'None')

def process_team(trojai_config: TrojaiConfig, g_drive: DriveIO, actor: Actor, active_leaderboards: Dict[str, Leaderboard], active_submission_managers: Dict[str, SubmissionManager]) -> None:

    for leaderboard_name, submission_manager in active_submission_managers.items():
        actor_submission_list = submission_manager.get_submissions_by_actor(actor)
        logging.debug('Found {} old submissions for actor {}.'.format(len(actor_submission_list), actor.name))
        for submission in actor_submission_list:
            # if the actor has any in flight submissions
            if submission.active_slurm_job_name is not None:
                logging.info('Found live submission "{}" from "{}"'.format(submission.g_file.name, submission.actor_name))

                if leaderboard_name not in active_leaderboards:
                    logging.warning('Leaderboard: {}, not found for submission: {}'.format(leaderboard_name, submission))
                    continue

                leaderboard = active_leaderboards[submission.leaderboard_name]
                submission.check(g_drive, actor, leaderboard, submission_manager, trojai_config.log_file_byte_limit)

    # look for any new submissions
    # This might modify the SubmissionManager instance
    logging.info('Done processing old/pending submission.')
    process_new_submission(trojai_config, g_drive, actor, active_leaderboards, active_submission_managers)


def main(trojai_config: TrojaiConfig) -> None:
    # load the instance of ActorManager from the serialized json file
    actor_manager = ActorManager.load_json(trojai_config)
    logging.debug('Loaded actor_manager from filepath: {}'.format(trojai_config.actors_filepath))
    logging.debug(actor_manager)

    # load the active leaderboards
    active_leaderboards = {}
    active_submission_managers = {}
    for leaderboard_name in trojai_config.active_leaderboard_names:
        leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
        active_leaderboards[leaderboard_name] = leaderboard
        submission_manager = SubmissionManager.load_json(leaderboard.submissions_filepath, leaderboard.name)
        active_submission_managers[leaderboard_name] = submission_manager
        logging.info('Leaderboard {}: Submissions Manger has {} actors and {} total submissions.'.format(leaderboard_name, submission_manager.get_number_actors(), submission_manager.get_number_submissions()))
        logging.info('Finished loading leaderboards and submission manager for: {}'.format(leaderboard_name))
        logging.debug(leaderboard)
        logging.debug(active_submission_managers[leaderboard_name])

    logging.info('Actor Manger has {} actors.'.format(len(actor_manager.get_keys())))


    g_drive = DriveIO(trojai_config.token_pickle_filepath)
    # Loop over actors, checking if there is a submission for each
    for key in actor_manager.get_keys():
        try:
            actor = actor_manager.get(key)
            logging.info('**************************************************')
            logging.info('Processing {}:'.format(actor.name))
            logging.info('**************************************************')
            process_team(trojai_config, g_drive, actor, active_leaderboards, active_submission_managers)
        except:
            msg = 'Exception processing actor "{}" loop:\n{}'.format(key, traceback.format_exc())
            logging.error(msg)

    # Check web-site updates
    update_html_pages(trojai_config, commit_and_push=True)

    # Write all updates to actors back to file
    logging.debug('Serializing updated actor_manger back to json.')
    actor_manager.save_json(trojai_config)

    logging.debug('Serializing updated submission_managers back to json.')
    # Should only have to save the submission manager. Leaderboard should be static
    for leaderboard_name, submission_manager in active_submission_managers.items():
        leaderboard = active_leaderboards[leaderboard_name]
        submission_manager.save_json(leaderboard.submissions_filepath)

    logging.info('Finished Check and Launch Actors')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check and Launch script for TrojAI challenge participants')
    parser.add_argument('--trojai-config-file', type=str,
                        help='The JSON file that describes trojai.', required=True)

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_file)

    # PidFile ensures that this script is only running once
    print('Attempting to acquire PID file lock.')
    lock_file = '/var/lock/trojai-lockfile'
    with open(lock_file, 'w') as f:
        try:
            fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print('  PID lock acquired')
            # make sure intermediate folders to the logfile exists
            parent_fp = os.path.dirname(trojai_config.log_filepath)
            if not os.path.exists(parent_fp):
                os.makedirs(parent_fp)
            # Add the log message handler to the logger
            handler = logging.handlers.RotatingFileHandler(trojai_config.log_filepath, maxBytes=100*1e6, backupCount=10) # 100MB
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                                handlers=[handler])
            # TODO: Remove this
            logging.getLogger().addHandler(logging.StreamHandler())

            logging.debug('PID file lock acquired in directory {}'.format(args.trojai_config_file))
            main(trojai_config)
        except OSError as e:
            print('check-and-launch was already running when called.')
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


