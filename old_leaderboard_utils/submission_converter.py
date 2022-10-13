from leaderboards.trojai_config import TrojaiConfig
from leaderboards.leaderboard import Leaderboard
from leaderboards.submission import SubmissionManager, Submission
from leaderboards.actor import ActorManager, Actor
from leaderboards.google_drive_file import GoogleDriveFile
from leaderboards import json_io
from leaderboards import time_utils

import os
import shutil


def convert_submission(args):
    old_prefix = args.old_prefix
    new_prefix = args.new_prefix
    copy_data = args.copy_data
    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    actor_manager = ActorManager.load_json(trojai_config)
    leaderboard = Leaderboard.load_json(trojai_config, args.leaderboard_name)

    prior_round_submission_manager = json_io.read(args.submission_filepath)
    current_submission_manager = SubmissionManager.load_json(leaderboard)

    data_split_name = args.data_split_name


    # Create new submission to be added to new format
    for actor_email, submission_list in prior_round_submission_manager['_SubmissionManager__submissions'].items():
        for old_submission in submission_list:
            old_actor_email = old_submission['actor']['email']
            old_actor_name = old_submission['actor']['name']
            try:
                actor = actor_manager.get(old_actor_email)
            except:
                print('Failed to get actor from submission for email: {}... Attempting prior emails'.format(old_actor_email))

                found_actors = []

                for cur_actor in actor_manager.get_actors():
                    for email in cur_actor.prior_emails:
                        if email == old_actor_email:
                            found_actors.append(cur_actor)
                            break

                if len(found_actors) > 1:
                    print('Found multiple actors with the same prior emails: {}'.format(found_actors))
                    continue
                elif len(found_actors) == 0:
                    print('Found no actors with the same prior email for: {}'.format(old_actor_email))
                    continue
                else:
                    actor = found_actors[0]
                    print('Found alternate actor {} for {}'.format(actor.name, old_actor_email))



            if actor is None:
                print('Failed to get actor from submission for email: {}'.format(old_actor_email))
                continue

            new_g_file = GoogleDriveFile(old_submission['file']['email'], old_submission['file']['name'], old_submission['file']['id'], time_utils.convert_epoch_to_iso(old_submission['file']['modified_epoch']))

            new_submission = Submission(new_g_file, actor, leaderboard, data_split_name, submission_epoch=old_submission['execution_epoch'])
            new_submission.actor_uuid = actor.uuid

            if 'execution_runtime' in old_submission:
                new_submission.execution_runtime = old_submission['execution_runtime']

            if 'model_execution_runtimes' in old_submission:
                new_submission.model_execution_runtimes = old_submission['model_execution_runtimes']

            new_submission.execution_epoch = old_submission['execution_epoch']
            new_submission.slurm_output_filename = old_submission['slurm_output_filename']
            new_submission.web_display_parse_errors = old_submission['web_display_parse_errors']
            new_submission.web_display_execution_errors = old_submission['web_display_execution_errors']
            new_submission.execution_results_dirpath = os.path.join(new_submission.actor_results_dirpath, new_submission.get_execute_time_str())

            current_submission_manager.add_submission(actor, new_submission)

            if copy_data:
                # Copy contents of submission to new location
                time_str = time_utils.convert_epoch_to_psudo_iso(old_submission['execution_epoch'])

                old_submission_container_dirpath = os.path.join(old_submission['global_submission_dirpath'], old_actor_name, time_str)
                old_submission_results_dirpath = os.path.join(old_submission['global_results_dirpath'], old_actor_name, time_str)
                new_submission_container_dirpath = new_submission.actor_submission_dirpath
                new_submission_results_dirpath = new_submission.execution_results_dirpath

                if old_prefix is not None and new_prefix is not None:
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

    current_submission_manager.save_json(leaderboard)








if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Converts an old submission into a new one for the multi-round leaderboard')
    parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the multi-round config', required=True)
    parser.add_argument('--submission-filepath', type=str, help='The filepath to the old round submission', required=True)
    parser.add_argument('--leaderboard-name', type=str, help='The name of the leaderboard that the submissions will be added too', required=True)
    parser.add_argument('--data-split-name', type=str, help='The data split name that the submissions should be added too', required=True)
    parser.add_argument('--old-prefix', type=str, help='The name of the old prefix that was in the old submission for renaming directory paths', default=None)
    parser.add_argument('--new-prefix', type=str, help='The name of the new prefix that is the current location of the old submissions for renaming directory paths', default=None)
    parser.add_argument('--copy-data', action='store_true', help='Whether to copy all data for the submissions')
    parser.set_defaults(func=convert_submission)

    args = parser.parse_args()
    args.func(args)