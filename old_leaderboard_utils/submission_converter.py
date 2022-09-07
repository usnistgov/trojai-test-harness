from leaderboards.trojai_config import TrojaiConfig
from leaderboards.leaderboard import Leaderboard
from leaderboards.submission import SubmissionManager, Submission
from leaderboards.actor import Actor
from leaderboards.google_drive_file import GoogleDriveFile
from actor_executor import submission
from actor_executor import time_utils

import os
import shutil


def convert_submission(args):
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    leaderboard = Leaderboard.load_json(trojai_config, args.leaderboard_name)

    prior_round_submission_manager = submission.SubmissionManager.load_json(args.submission_filepath)
    current_submission_manager = SubmissionManager.load_json(leaderboard.submissions_filepath, leaderboard.name)

    data_split_name = args.data_split_name


    # Create new submission to be added to new format
    for actor_email, submission_list in prior_round_submission_manager.submissions().items():
        for old_submission in submission_list:
            actor = Actor(trojai_config, old_submission.actor['email'], old_submission.actor['name'], old_submission.actor['poc_email'], 'normal', reset=False)

            new_g_file = GoogleDriveFile(old_submission.file.email, old_submission.file.name, old_submission.file.id, time_utils.convert_epoch_to_iso(old_submission.file.modified_epoch))

            new_submission = Submission(new_g_file, actor, leaderboard, data_split_name)
            new_submission.actor_email = actor_email
            new_submission.actor_name = old_submission.actor['name']
            new_submission.execution_runtime = old_submission.execution_runtime
            new_submission.model_execution_runtimes = old_submission.model_execution_runtimes
            new_submission.execution_epoch = old_submission.execution_epoch
            new_submission.slurm_output_filename = old_submission.slurm_output_filename
            new_submission.web_display_parse_errors = old_submission.web_display_parse_errors
            new_submission.web_display_execution_errors = old_submission.web_display_execution_errors

            current_submission_manager.add_submission(actor, new_submission)

            # Copy contents of submission to new location
            time_str = time_utils.convert_epoch_to_psudo_iso(old_submission.execution_epoch)

            old_submission_container_dirpath = os.path.join(old_submission.global_submission_dirpath, actor.name, time_str)
            old_submission_results_dirpath = os.path.join(old_submission.global_results_dirpath, actor.name, time_str)
            new_submission_container_dirpath = os.path.join(new_submission.actor_submission_dirpath, time_str)
            new_submission_results_dirpath = os.path.join(new_submission.actor_results_dirpath, time_str)

            old_prefix = '/mnt/trojainas/'
            new_prefix = '/home/tjb3/old-te/'
            old_submission_container_dirpath = old_submission_container_dirpath.replace(old_prefix, new_prefix)
            old_submission_results_dirpath = old_submission_results_dirpath.replace(old_prefix, new_prefix)

            if not os.path.exists(new_submission_container_dirpath):
                os.makedirs(new_submission_container_dirpath)

            if not os.path.exists(new_submission_results_dirpath):
                os.makedirs(new_submission_results_dirpath)

            if os.path.exists(old_submission_container_dirpath):
                print('Copying old submission for {}:{} into new'.format(actor.name, time_str))
                shutil.copytree(old_submission_container_dirpath, new_submission_container_dirpath, dirs_exist_ok=True)
            else:
                print('Warning, unable to locate old submissions dirpath: {}'.format(old_submission_container_dirpath))


            if os.path.exists(old_submission_results_dirpath):
                print('Copying old submission for {}:{} results into new'.format(actor.name, time_str))
                shutil.copytree(old_submission_results_dirpath, new_submission_results_dirpath, dirs_exist_ok=True)
            else:
                print('Warning, unable to locate old submissions results dirpath: {}'.format(old_submission_results_dirpath))

    current_submission_manager.save_json(leaderboard.submissions_filepath)








if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Converts an old submission into a new one for the multi-round leaderboard')
    parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the multi-round config', required=True)
    parser.add_argument('--submission-filepath', type=str, help='The filepath to the old round submission', required=True)
    parser.add_argument('--leaderboard-name', type=str, help='The name of the leaderboard that the submissions will be added too', required=True)
    parser.add_argument('--data-split-name', type=str, help='The data split name that the submissions should be added too', required=True)
    parser.set_defaults(func=convert_submission)

    args = parser.parse_args()
    args.func(args)