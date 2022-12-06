from leaderboards.trojai_config import TrojaiConfig
from leaderboards.leaderboard import Leaderboard
from leaderboards.submission import SubmissionManager, Submission
from leaderboards.actor import ActorManager, Actor
from leaderboards.google_drive_file import GoogleDriveFile

from leaderboards import json_io
from leaderboards import time_utils

import os
import shutil

def create_submission_json(args):
    old_round_dirpath = args.old_holdout_dirpath
    copy_data = args.copy_data

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    actor_manager = ActorManager.load_json(trojai_config)
    leaderboard = Leaderboard.load_json(trojai_config, args.leaderboard_name)
    data_split_name = args.data_split_name


    results_dirpath = os.path.join(old_round_dirpath, 'results')
    submissions_dirpath = os.path.join(old_round_dirpath, 'submissions')

    prior_round_submission_manager = json_io.read(args.old_test_submission_filepath)
    current_submission_manager = SubmissionManager.load_json(leaderboard)

    prior_round_submissions_dict = prior_round_submission_manager['_SubmissionManager__submissions']

    for actor_name in os.listdir(results_dirpath):
        # Get the actor entry
        actor = actor_manager.get_from_name(actor_name)

        submission_list = None

        # find the submissions from
        if actor.email in prior_round_submissions_dict:
            submission_list = prior_round_submissions_dict[actor.email]
        else:
            print('Failed to find {}, checking prior emails'.format(actor.email))
            for actor_email in actor.prior_emails:
                if actor_email in prior_round_submissions_dict:
                    submission_list = prior_round_submissions_dict[actor.email]
                    break

        if submission_list is None:
            print('Failed to find submissions for {}'.format(actor_name))
            continue


        actor_submissions_dirpath = os.path.join(submissions_dirpath, actor_name)

        for execution_timestamp in os.listdir(actor_submissions_dirpath):
            execution_epoch = time_utils.convert_to_epoch_from_psudo(execution_timestamp)

            # Find submission based on submission epoch
            test_submission = None
            for submission in submission_list:
                if submission['execution_epoch'] == execution_epoch:
                    test_submission = submission
                    break

            if test_submission is None:
                print('Failed to find submission for {} for actor {}'.format(execution_timestamp, actor_name))
                continue

            new_g_file = GoogleDriveFile(test_submission['file']['email'], test_submission['file']['name'], test_submission['file']['id'], time_utils.convert_epoch_to_iso(test_submission['file']['modified_epoch']))

            new_submission = Submission(new_g_file, actor, leaderboard, data_split_name, submission_epoch=execution_epoch)
            new_submission.actor_uuid = actor.uuid

            new_submission.execution_epoch = execution_epoch
            new_submission.slurm_output_filename = test_submission['slurm_output_filename']
            new_submission.web_display_parse_errors = test_submission['web_display_parse_errors']
            new_submission.web_display_execution_errors = test_submission['web_display_execution_errors']
            new_submission.execution_results_dirpath = os.path.join(new_submission.actor_results_dirpath, new_submission.get_execute_time_str())

            current_submission_manager.add_submission(actor, new_submission)

            if copy_data:
                old_submission_container_dirpath = os.path.join(submissions_dirpath, actor_name, execution_timestamp)
                old_submission_results_dirpath = os.path.join(results_dirpath, actor_name, execution_timestamp)
                new_submission_container_dirpath = new_submission.actor_submission_dirpath
                new_submission_results_dirpath = new_submission.execution_results_dirpath

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


    current_submission_manager.save_json(leaderboard)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create submissions json based on directory of old round data.')
    parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the multi-round config', required=True)
    parser.add_argument('--old-holdout-dirpath', type=str, help='The directory path to the old round data', required=True)
    parser.add_argument('--old-test-submission-filepath', type=str, help='The filepath to the test submissions from the old round')
    parser.add_argument('--leaderboard-name', type=str, help='The name of the leaderboard that the submissions will be added too', required=True)
    parser.add_argument('--data-split-name', type=str, help='The data split name that the submissions should be added too', required=True)

    parser.add_argument('--copy-data', action='store_true', help='Whether to copy all data for the submission')
    parser.set_defaults(func=create_submission_json)
    args = parser.parse_args()

    args.func(args)
