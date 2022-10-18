# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import logging
import logging.handlers
import shutil
import subprocess
import numpy as np
import time

from leaderboards import time_utils
from leaderboards.trojai_config import TrojaiConfig
from leaderboards.submission import Submission, SubmissionManager
from leaderboards.actor import ActorManager
from leaderboards.leaderboard import Leaderboard
from leaderboards.metrics import Metric


def main(trojai_config: TrojaiConfig, container_leaderboard_name: str, container_data_split_name: str,
         execution_leaderboard_name: str,  execution_data_split_name: str, execution_submission_filepath: str,
         metric_name: str, target_metric_value: float, custom_home:str, custom_scratch:str, python_env_filepath:str, skip_existing_submissions: bool=False, execution_submission_exists_okay=False,
         team_names: list = [], provenance_name: str='custom-launch', slurm_queue_name='heimdall',
         custom_slurm_options: list=[]) -> None:
    actor_manager = ActorManager.load_json(trojai_config)
    container_leaderboard = Leaderboard.load_json(trojai_config, container_leaderboard_name)
    container_submission_manager = SubmissionManager.load_json(container_leaderboard)

    execution_leaderboard = Leaderboard.load_json(trojai_config, execution_leaderboard_name)
    custom_submission_manager_filepath = execution_submission_filepath #os.path.join(os.path.dirname(execution_leaderboard.submissions_filepath), execution_submission_filename)

    if not execution_submission_exists_okay:
        if os.path.exists(custom_submission_manager_filepath):
            raise RuntimeError('Submission manager: {} already exists, disable exists_okay or specify a new execution submission filename'.format(custom_submission_manager_filepath))

    execution_submission_manager = SubmissionManager.load_json_custom(custom_submission_manager_filepath, execution_leaderboard_name)

    logging.info('Starting finding submissions from leaderboard {} for data split {} to execute on leaderboard {} with datasplit {}'.format(container_leaderboard_name, container_data_split_name, execution_leaderboard_name, execution_data_split_name))


    # Gather submissions based on metric
    search_actors = []
    for actor in actor_manager.get_actors():
        if len(team_names) == 0:
            search_actors.append(actor)
        elif actor.name in team_names:
            search_actors.append(actor)

    if len(search_actors) == 0:
        raise RuntimeError('Failed to find any actors from actor list: {}'.format(team_names))

    valid_submissions = []

    for actor in search_actors:
        submissions = container_submission_manager.gather_submissions(container_leaderboard, container_data_split_name, metric_name, target_metric_value, actor)
        valid_submissions.extend(submissions)

    existing_submission_manager = SubmissionManager.load_json(execution_leaderboard)

    logging.info('Found {} submissions that meet the target metric criteria {} for metric {}'.format(len(valid_submissions), target_metric_value, metric_name))

    # Launch submissions
    for submission in valid_submissions:
        actor = actor_manager.get_from_uuid(submission.actor_uuid)
        new_submission = Submission(submission.g_file, actor, execution_leaderboard, execution_data_split_name, provenance_name, submission.submission_epoch, slurm_queue_name, container_leaderboard)

        if skip_existing_submissions:
            actor_existing_submissions = existing_submission_manager.get_submissions_by_actor(actor)
            found_existing = False
            for submission in actor_existing_submissions:
                if submission.actor_submission_dirpath == new_submission.actor_submission_dirpath and submission.actor_results_dirpath == new_submission.actor_results_dirpath:
                    # Check for results
                    if new_submission.execution_results_dirpath is not None:
                        logging.info('Found matching submission for: {}, skip existing submissions is enabled'.format(actor.name))
                        found_existing = True
                        break

            if found_existing:
                continue


        time.sleep(1)
        exec_epoch = time_utils.get_current_epoch()
        new_submission.execute(actor, trojai_config, exec_epoch, execute_local=True, custom_home_dirpath=custom_home, custom_scratch_dirpath=custom_scratch, custom_slurm_options=custom_slurm_options, custom_python_env_filepath=python_env_filepath)
        execution_submission_manager.add_submission(actor, new_submission)


    execution_submission_manager.save_json_custom(custom_submission_manager_filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executes holdout data on actors that meet criteria")

    parser.add_argument('--trojai-config-filepath', type=str,
                        help='Filepath trojai config file',
                        default='trojai-config.json')

    parser.add_argument('--container-leaderboard-name', type=str, help='The name of the leaderboard where singularity containers that you want to execute live', required=True)
    parser.add_argument('--container-data-split-name', type=str, help='The name of the data split within the container leaderboard to find containers to execute', required=True)

    parser.add_argument('--execution-leaderboard-name', type=str, help='The name of the leaderboard to execute against, containing the dataset you want to use', required=True)
    parser.add_argument('--execution-data-split-name', type=str, help='The name of the data split within the leaderboard to execute, indicating which dataset to use', required=True)

    parser.add_argument('--execution-submission-filepath', type=str, help='The filepath to submission file that will be created', default='./custom-launch-submissions.json')

    parser.add_argument('--metric-name', type=str, help='The name of the metric to use, as seen on the leaderboard', default='Cross Entropy')
    parser.add_argument('--target-metric-value', type=float, help='The target value that containers must meet to execute', default='0.5')

    parser.add_argument('--custom-home', type=str, help='The directory to use when executing singularity containers', required=True)
    parser.add_argument('--custom-scratch', type=str, help='The scratch directory to use when executing singularity containers', required=True)
    parser.add_argument('--skip-existing', action='store_true', help='Whether to skip any existing submissions to avoid added compute')

    parser.add_argument('--execution-submission-exists-okay', action='store_true', help='Indicates that it is okay if the output submission file exists, which will load prior submissions and add to them.')

    parser.add_argument('--team-names', nargs='*', help='The names of the teams to evaluate, default will use all teams', default=[])
    parser.add_argument('--provenance-name', type=str, help='The provenance name for this run', default='custom-launch')
    parser.add_argument('--slurm-partition-name', type=str, help='The name of the slurm partition to launch into')
    parser.add_argument('--custom-slurm-options', nargs='*', help='Options to pass into slurm', default=['--gres=gpu:1'])
    parser.add_argument('--python-filepath', type=str, help='The filepath to the python executable to be used within the cluster', required=True)

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)

    # handler = logging.handlers.RotatingFileHandler(trojai_config.log_filepath, maxBytes=100 * 1e6, backupCount=10)  # 100MB
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

    # logging.getLogger().addHandler(logging.StreamHandler())

    main(trojai_config, args.container_leaderboard_name, args.container_data_split_name,
         args.execution_leaderboard_name, args.execution_data_split_name, args.execution_submission_filepath,
         args.metric_name, args.target_metric_value, args.custom_home, args.custom_scratch, args.python_filepath, args.skip_existing,
         args.execution_submission_exists_okay, args.team_names, args.provenance_name, args.slurm_partition_name, args.custom_slurm_options)


