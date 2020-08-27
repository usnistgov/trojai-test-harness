import os
import logging
import logging.handlers
import shutil
import subprocess

import time_utils
from config import Config, HoldoutConfig
from submission import Submission, SubmissionManager


def main(round_config_path: str, round_config: Config, holdout_config_path: str, holdout_config: HoldoutConfig, execute_team_name: str) -> None:

    submission_manager = SubmissionManager.load_json(round_config.submissions_json_file)
    logging.debug('Loaded submission_manager from filepath: {}'.format(round_config.submissions_json_file))
    logging.debug(submission_manager)

    if not os.path.exists(holdout_config.results_dir):
        logging.info('Creating results directory: {}'.format(holdout_config.results_dir))
        os.makedirs(holdout_config.results_dir)
    if not os.path.exists(holdout_config.submission_dir):
        logging.info('Creating submissions directory: {}'.format(holdout_config.submission_dir))
        os.makedirs(holdout_config.submission_dir)

    # Gather submissions based on criteria
    # Key = actor email, value = list of submissions that meets min loss criteria
    holdout_execution_submissions = submission_manager.gather_submissions(holdout_config.min_loss_criteria, execute_team_name)

    for actor_email in holdout_execution_submissions.keys():
        # process list of submissions for the actor
        submissions = holdout_execution_submissions[actor_email]

        logging.info("Submitting {} submissions for actor email {}".format(len(submissions), actor_email))

        for submission in submissions:
            time_str = time_utils.convert_epoch_to_psudo_iso(submission.execution_epoch)
            existing_actor_submission_filepath = os.path.join(submission.global_submission_dirpath, submission.actor.name, time_str, submission.file.name)

            if not os.path.exists(existing_actor_submission_filepath):
                logging.error('Unable to find {}, cannot execute submission without container.'.format(existing_actor_submission_filepath))
                continue

            container_submission_dir = os.path.join(holdout_config.submission_dir, submission.actor.name, time_str)
            holdout_actor_submission_filepath = os.path.join(container_submission_dir, submission.file.name)
            if not os.path.exists(container_submission_dir):
                logging.info('Creating directory to hold container image. {}'.format(container_submission_dir))
                os.makedirs(container_submission_dir)

            # Copy existing submission into holdout record
            logging.info('Copying container from {} to {}.'.format(existing_actor_submission_filepath, holdout_actor_submission_filepath))
            shutil.copyfile(existing_actor_submission_filepath, holdout_actor_submission_filepath)

            holdout_actor_results_dirpath = os.path.join(holdout_config.results_dir, submission.actor.name, time_str)
            if not os.path.exists(holdout_actor_results_dirpath):
                logging.debug('Creating result directory: {}'.format(holdout_actor_results_dirpath))
                os.makedirs(holdout_actor_results_dirpath)
            else:
                logging.debug('Result directory {} already exists, recreating it to purge old results'.format(holdout_actor_results_dirpath))
                shutil.rmtree(holdout_actor_results_dirpath)
                os.makedirs(holdout_actor_results_dirpath)

            slurm_output_filename = submission.actor.name + ".holdout.log.txt"
            slurm_job_name = submission.actor.name
            slurm_output_filepath = os.path.join(holdout_actor_results_dirpath, slurm_output_filename)

            v100_slurm_queue = 'control'

            cmd_str_list = ['sbatch', "--partition", v100_slurm_queue, "-n", "1", ":", "--partition", holdout_config.slurm_queue,
                            "--gres=gpu:1", "-J", slurm_job_name, "--parsable", "-o", slurm_output_filepath,
                            holdout_config.slurm_script, submission.actor.name, holdout_actor_submission_filepath, holdout_actor_results_dirpath, round_config_path, holdout_config_path, holdout_config.python_executor_script]

            logging.info('Launching holdout computation for actor: {}'.format(submission.actor.name))
            logging.info('\tES Cross entropy loss: {}'.format(submission.cross_entropy))
            logging.info('\tSubmission container: {}'.format(holdout_actor_submission_filepath))
            logging.info('\tHoldout result dir: {}'.format(holdout_actor_results_dirpath))
            logging.info('\tRound config: {}'.format(round_config_path))
            logging.info('\tHoldout config: {}'.format(holdout_config_path))

            out = subprocess.Popen(cmd_str_list,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            stdout, stderr = out.communicate()

            # Check if there are no errors reported from sbatch
            if stderr == b'':
                job_id = int(stdout.strip())
                logging.info("Slurm job executed with job id: {}".format(job_id))
            else:
                logging.error("The slurm script: {} resulted in errors {}".format(holdout_config.slurm_script, stderr))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executes holdout data on actors that meet criteria")

    parser.add_argument('--holdout-config-file', type=str,
                        help='The JSON file that describes the holdout execution',
                        default='holdout-config.json')

    parser.add_argument('--execute-team-name', type=str,
                        help='Executes the best model from team name',
                        default=None)

    args = parser.parse_args()

    holdout_config = HoldoutConfig.load_json(args.holdout_config_file)
    round_config = Config.load_json(holdout_config.round_config_filepath)
    execute_team_name = args.execute_team_name

    handler = logging.handlers.RotatingFileHandler(holdout_config.log_file, maxBytes=100*1e6, backupCount=10) # 100MB
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[handler])

    logging.info('Starting parsing for holdout execution')
    logging.info(holdout_config)
    logging.info(round_config)
    main(holdout_config.round_config_filepath, round_config, args.holdout_config_file, holdout_config, execute_team_name)
