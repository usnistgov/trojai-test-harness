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

import warnings



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

    # Search for entries that contain a valid leaderboard name and dataset split
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

        # Expected format is leaderboard-name_data-split-name_container-name.simg
        if len(filename_split) <= 2:
            logging.info('File {} from actor {} did not have expected format'.format(filename, actor.name))
            actor.general_file_status = 'Shared File Error (format)'
            has_general_errors = True
            continue

        leaderboard_name = filename_split[0]
        data_split_name = filename_split[1]

        # check if valid leaderboard
        if leaderboard_name not in active_leaderboards.keys():
            logging.info('File {} from actor {} did not have a valid leaderboard name: {}'.format(filename, actor.name, leaderboard_name))
            if not has_general_errors:
                actor.general_file_status = 'Shared File Error (leaderboard name)'
                has_general_errors = True
            continue

        leaderboard = active_leaderboards[leaderboard_name]

        # Check if valid data split name
        if not leaderboard.can_submit_to_dataset(data_split_name):
            logging.info('File {} from actor {} did not have a valid data split name: {}'.format(filename, actor.name, data_split_name))
            if not has_general_errors:
                actor.general_file_status = 'Shared File Error (data split name)'
                has_general_errors = True
            continue

        key = '{}_{}'.format(leaderboard_name, data_split_name)
        if key not in valid_submissions.keys():
            logging.info('Unknown leaderboard key when adding valid submissions: {}'.format(key))
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

        # Check if actor already has a job waiting to be processed (may be in queue)
        if submission_manager.has_active_submission(actor, data_split_name):
            logging.info('Detected another submission for {}, named {}, but an active submission is in progress'.format(actor.name, key))
            continue

        if len(g_file_list) == 0:
            logging.info('Actor {} does not have a submission from email {} for leaderboard {} and data split {}.'.format(actor.name, actor.email, leaderboard_name, data_split_name))
            actor.update_file_status(leaderboard_name, data_split_name, 'None')
            actor.update_job_status(leaderboard_name, data_split_name, 'None')

        if len(g_file_list) > 1:
            logging.warning('Actor {} shared {} files from email {} for leaderboard {} and data split {}'.format(actor.name, len(g_file_list), actor.email, leaderboard_name, data_split_name))
            actor.update_file_status(leaderboard_name, data_split_name, 'Multiple files shared')
            actor.update_job_status(leaderboard_name, data_split_name, 'None')

        if len(g_file_list) == 1:
            g_file = g_file_list[0]
            logging.info('Detected submission from actor {}: {} for leaderboard {} and data split {}'.format(actor.name, g_file.name, leaderboard_name, data_split_name))
            actor.update_file_status(leaderboard_name, data_split_name, 'Ok')

            # Sleep for 1 second to have distinct execute epochs
            time.sleep(1)

            check_epoch = time_utils.get_current_epoch()

            if not actor.can_submit_time_window(leaderboard_name, data_split_name, leaderboard.get_submission_window_time(data_split_name), check_epoch) and int(g_file.modified_epoch) != int(actor.get_last_file_epoch(leaderboard_name, data_split_name)):
                logging.info('Team {} timeout window has not elapsed. check_epoch: {}, last_submission_epoch: {}, leaderboards: {}, data split: {}'.format(actor.name, check_epoch, actor.get_last_submission_epoch(leaderboard_name, data_split_name), leaderboard_name, data_split_name))
                actor.update_job_status(leaderboard_name, data_split_name, 'Awaiting Timeout')
            else:
                if int(g_file.modified_epoch) != int(actor.get_last_file_epoch(leaderboard_name, data_split_name)):
                    if not submission_manager.has_submission_file_id(actor, g_file.modified_epoch):
                        logging.info('Submission timestamp is different .... EXECUTING; new file name: {}, new file epoch: {}, last file epoch: {}'.format(g_file.name, g_file.modified_epoch, actor.get_last_file_epoch(leaderboard_name, data_split_name)))
                        submission = Submission(g_file, actor, leaderboard, data_split_name)
                        submission_manager.add_submission(actor, submission)
                        logging.info('Added submission file name "{}" to manager from email "{}"'.format(submission.g_file.name, actor.email))
                        exec_epoch = time_utils.get_current_epoch()
                        submission.execute(actor, trojai_config, exec_epoch)
                    else:
                        logging.info(
                            'Submission found is the same within one of the submissions already in the submission manager for team {}; new file name: {}, new file epoch: {}'.format(
                                actor.name, g_file.name, g_file.modified_epoch))
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
                logging.info('Found live submission "{}" from "{}"'.format(submission.g_file.name, actor.name))

                if leaderboard_name not in active_leaderboards:
                    logging.warning('Leaderboard: {}, not found for submission: {}'.format(leaderboard_name, submission))
                    continue

                leaderboard = active_leaderboards[submission.leaderboard_name]
                submission.check(trojai_config, g_drive, actor, leaderboard, submission_manager, trojai_config.log_file_byte_limit)

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
    archive_leaderboards = {}
    archive_submission_managers = {}

    for leaderboard_name in trojai_config.active_leaderboard_names:
        leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
        if leaderboard is None:
            continue

        # Check for any new instance data during development
        leaderboard.check_instance_data(trojai_config)

        active_leaderboards[leaderboard_name] = leaderboard
        submission_manager = SubmissionManager.load_json(leaderboard)
        active_submission_managers[leaderboard_name] = submission_manager
        logging.info('Leaderboard {}: Submissions Manger has {} actors and {} total submissions.'.format(leaderboard_name, submission_manager.get_number_actors(), submission_manager.get_number_submissions()))
        logging.info('Finished loading leaderboards and submission manager for: {}'.format(leaderboard_name))
        logging.debug(leaderboard)
        logging.debug(active_submission_managers[leaderboard_name])

    for leaderboard_name in trojai_config.archive_leaderboard_names:
        leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
        if leaderboard is None:
            continue
        archive_leaderboards[leaderboard_name] = leaderboard
        archive_submission_managers[leaderboard_name] = SubmissionManager.load_json(leaderboard)
        logging.info('Archived Leaderboard {} loaded'.format(leaderboard_name))

    logging.info('Actor Manger has {} actors.'.format(len(actor_manager.get_keys())))

    g_drive = DriveIO(trojai_config.token_pickle_filepath)
    # Loop over actors, checking if there is a submission for each
    for actor in actor_manager.get_actors():
        try:
            logging.info('**************************************************')
            logging.info('Processing {}:'.format(actor.name))
            logging.info('**************************************************')
            process_team(trojai_config, g_drive, actor, active_leaderboards, active_submission_managers)
        except:
            msg = 'Exception processing actor "{}" loop:\n{}'.format(actor.name, traceback.format_exc())
            logging.error(msg)

    logging.info('Saving actors and submission managers prior to checking new metrics')
    # Write all updates to actors back to file
    logging.debug('Serializing updated actor_manger back to json.')
    actor_manager.save_json(trojai_config)

    logging.debug('Serializing updated submission_managers back to json.')
    # Should only have to save the submission manager. Leaderboard should be static
    for leaderboard_name, submission_manager in active_submission_managers.items():
        leaderboard = active_leaderboards[leaderboard_name]
        submission_manager.save_json(leaderboard)

    for leaderboard_name, submission_manager in archive_submission_managers.items():
        leaderboard = archive_leaderboards[leaderboard_name]
        submission_manager.save_json(leaderboard)

    logging.info('Checking for new/missing metrics')
    # Check to see if we need to compute any new/missing metrics
    for leaderboard_name, leaderboard in active_leaderboards.items():
        if leaderboard.check_for_missing_metrics:
            submission_manager = active_submission_managers[leaderboard.name]
            submission_manager.check_for_new_metrics(leaderboard, actor_manager, g_drive)
            submission_manager.save_json(leaderboard)
        else:
            logging.info('Skipping check new/missing for {}'.format(leaderboard_name))

    for leaderboard_name, leaderboard in archive_leaderboards.items():
        if leaderboard.check_for_missing_metrics:
            submission_manager = archive_submission_managers[leaderboard.name]
            submission_manager.check_for_new_metrics(leaderboard, actor_manager, g_drive)
            submission_manager.save_json(leaderboard)
        else:
            logging.info('Skipping check new/missing for {}'.format(leaderboard_name))

    # Apply summary updates
    cur_epoch = time_utils.get_current_epoch()
    summary_html_plots = []
    if trojai_config.can_apply_summary_updates(cur_epoch):
        logging.info('Applying summary metric updates')
        if not os.path.exists(trojai_config.summary_metrics_dirpath):
            os.makedirs(trojai_config.summary_metrics_dirpath)

        trojai_summary_folder_id = g_drive.create_folder('trojai_summary_plots')

        # Run global metric updates for active leaderboards
        for leaderboard_name, leaderboard in active_leaderboards.items():
            leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)

            # Upload summary schema CSV
            summary_schema_csv_filepath = leaderboard.get_summary_schema_csv_filepath(trojai_config)
            if os.path.exists(summary_schema_csv_filepath):
                g_drive.upload(summary_schema_csv_filepath, folder_id=trojai_summary_folder_id)

            submission_manager = active_submission_managers[leaderboard_name]

            leaderboard.generate_metadata_csv(overwrite_csv=True)
            submission_manager.generate_round_results_csv(leaderboard, actor_manager, overwrite_csv=False)

            g_drive.upload(leaderboard.summary_metadata_csv_filepath, trojai_summary_folder_id)
            g_drive.upload(leaderboard.summary_results_csv_filepath, trojai_summary_folder_id)

            metadata_df = leaderboard.load_metadata_csv_into_df()
            results_df = leaderboard.load_summary_results_csv_into_df()

            for data_split_name in leaderboard.get_all_data_split_names():
                if data_split_name == 'sts':
                    continue

                leaderboard_folder_id = g_drive.create_folder('{}_{}'.format(leaderboard_name, data_split_name), trojai_summary_folder_id)

                # Subset metadata and results dfs
                subset_metadata_df = metadata_df[metadata_df['data_split'] == data_split_name]
                if results_df is not None:
                    subset_results_df = results_df[results_df['data_split'] == data_split_name]
                else:
                    subset_results_df = None


                # Process summary metrics
                for summary_metric in leaderboard.summary_metrics:
                    if subset_results_df is None:
                        continue
                        
                    output_files = summary_metric.compute_and_write_data(leaderboard_name, data_split_name, subset_metadata_df, subset_results_df, trojai_config.summary_metrics_dirpath)

                    if summary_metric.shared_with_collaborators:
                        for file in output_files:
                            g_drive.upload(file, leaderboard_folder_id)

                    if summary_metric.add_to_html:
                        summary_html_plots.extend(output_files)

        # Run global metric updates for archive leaderboards if they don't exist
        for leaderboard_name, leaderboard in archive_leaderboards.items():
            submission_manager = archive_submission_managers[leaderboard.name]

            leaderboard.generate_metadata_csv(overwrite_csv=False)
            submission_manager.generate_round_results_csv(leaderboard, actor_manager, overwrite_csv=False)

            g_drive.upload(leaderboard.summary_metadata_csv_filepath, trojai_summary_folder_id)
            g_drive.upload(leaderboard.summary_results_csv_filepath, trojai_summary_folder_id)

            metadata_df = leaderboard.load_metadata_csv_into_df()
            results_df = leaderboard.load_summary_results_csv_into_df()

            for data_split_name in leaderboard.get_all_data_split_names():
                if data_split_name == 'sts':
                    continue

                leaderboard_folder_id = g_drive.create_folder('{}_{}'.format(leaderboard_name, data_split_name), trojai_summary_folder_id)

                # Subset metadata and results dfs
                subset_metadata_df = metadata_df[metadata_df['data_split'] == data_split_name]
                subset_results_df = results_df[results_df['data_split'] == data_split_name]

                for summary_metric in leaderboard.summary_metrics:
                    output_files = summary_metric.compute_and_write_data(leaderboard_name, data_split_name, subset_metadata_df, subset_results_df, trojai_config.summary_metrics_dirpath)

                    if summary_metric.shared_with_collaborators:
                        for file in output_files:
                            g_drive.upload(file, leaderboard_folder_id)

                    if summary_metric.add_to_html:
                        summary_html_plots.extend(output_files)

        # Share summary metrics
        g_drive.remove_all_sharing_permissions(trojai_summary_folder_id)
        for email in trojai_config.summary_metric_email_addresses:
            g_drive.share(trojai_summary_folder_id, email)

    # Check web-site updates
    logging.info('Updating HTML pages')
    update_html_pages(trojai_config, actor_manager, active_leaderboards, active_submission_managers, archive_leaderboards, archive_submission_managers, commit_and_push=trojai_config.commit_and_push_html, g_drive=g_drive)

    # Write all updates to actors back to file
    logging.debug('Serializing updated actor_manger back to json.')
    actor_manager.save_json(trojai_config)

    logging.debug('Serializing updated submission_managers back to json.')
    # Should only have to save the submission manager. Leaderboard should be static
    for leaderboard_name, submission_manager in active_submission_managers.items():
        leaderboard = active_leaderboards[leaderboard_name]
        submission_manager.save_json(leaderboard)

    for leaderboard_name, submission_manager in archive_submission_managers.items():
        leaderboard = archive_leaderboards[leaderboard_name]
        submission_manager.save_json(leaderboard)

    logging.info('Finished Check and Launch Actors, total g_drive API requests: {}'.format(g_drive.request_count))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", module="matplotlib\..*")
    import argparse

    parser = argparse.ArgumentParser(description='Check and Launch script for TrojAI challenge participants')
    parser.add_argument('--trojai-config-filepath', type=str,
                        help='The JSON file that describes trojai.', required=True)

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

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
            # Enable when debugging
            # logging.getLogger().addHandler(logging.StreamHandler())

            logging.debug('PID file lock acquired in directory {}'.format(args.trojai_config_filepath))
            main(trojai_config)
        except OSError as e:
            print('check-and-launch was already running when called. {}'.format(e))
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


