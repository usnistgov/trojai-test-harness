import os
import logging
import logging.handlers
import subprocess

import time_utils
from config import Config
from holdout_config import HoldoutConfig
from submission import Submission, SubmissionManager


def main(round_config_path:str, round_config: Config, holdout_config_path: str, holdout_config: HoldoutConfig) -> None:
    submission_manager = SubmissionManager.load_json(round_config.submissions_json_file)
    logging.debug('Loaded submission_manager from filepath: {}'.format(round_config.submissions_json_file))
    logging.debug(submission_manager)

    # Gather submissions based on criteria
    min_loss_criteria = holdout_config.min_loss_criteria

    # Key = actor, value = submission that is best that meets criteria
    holdout_execution_submissions = submission_manager.gather_submissions(holdout_config.min_loss_criteria)

    for actor_email in holdout_execution_submissions.keys():
        submission = holdout_execution_submissions[actor_email]
        time_str = time_utils.convert_epoch_to_psudo_iso(submission.execution_epoch)
        actor_submission_filepath = os.path.join(submission.global_submission_dirpath, submission.actor.name, time_str, submission.file.name)

        result_dirpath = os.path.join(holdout_config.holdout_result_dir, submission.actor.name)
        if not os.path.exists(result_dirpath):
            logging.debug('Creating result directory: {}'.format(result_dirpath))
            os.makedirs(result_dirpath)

        slurm_output_filename = submission.actor.name + ".es.log.txt"
        slurm_job_name = submission.actor.name
        slurm_output_filepath = os.path.join(result_dirpath, slurm_output_filename)

        v100_slurm_queue = 'control'

        cmd_str_list = ['sbatch', "--partition", v100_slurm_queue, "-n", "1", ":", "--partition", holdout_config.slurm_queue,
                        "--gres=gpu:1", "-J", slurm_job_name, "--parsable", "-o", slurm_output_filepath,
                        holdout_config.slurm_script, submission.actor.name, actor_submission_filepath, result_dirpath, round_config_path,
                        submission.actor.email, holdout_config_path, holdout_config.python_executor_script]
        logging.info('launching sbatch command: "{}"'.format(' '.join(cmd_str_list)))
        print(cmd_str_list)
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

    args = parser.parse_args()

    holdout_config = HoldoutConfig.load_json(args.holdout_config_file)
    round_config = Config.load_json(holdout_config.round_config_filepath)

    handler = logging.handlers.RotatingFileHandler(holdout_config.log_file, maxBytes=100*1e6, backupCount=10) # 100MB
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[handler])

    logging.info('Starting parsing for holdout execution')
    main(holdout_config.round_config_filepath, round_config, args.holdout_config_file, holdout_config)
