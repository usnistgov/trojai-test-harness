# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import logging
import logging.handlers
import shutil
import subprocess
import numpy as np

from actor_executor import time_utils
from actor_executor.config import Config
from actor_executor.submission import Submission, SubmissionManager


def main(config_filepath: str, config: Config, execute_team_name: str) -> None:

    submission_manager = SubmissionManager.load_json(config.submissions_json_file)
    logging.debug('Loaded submission_manager from filepath: {}'.format(config.submissions_json_file))
    logging.debug(submission_manager)

    if not os.path.exists(config.results_dir):
        logging.info('Creating results directory: {}'.format(config.results_dir))
        os.makedirs(config.results_dir)
    if not os.path.exists(config.submission_dir):
        logging.info('Creating submission_dir directory: {}'.format(config.submission_dir))
        os.makedirs(config.submission_dir)

    # Gather submissions based on criteria
    # Key = actor email, value = list of submissions that meets min loss criteria
    if config.loss_criteria is None or not np.isfinite(config.loss_criteria):
        logging.error('Loss criteria "{}" must be a valid float.'.format(config.loss_criteria))
        raise RuntimeError('Loss criteria "{}" must be a valid float.'.format(config.loss_criteria))
    holdout_execution_submissions = submission_manager.gather_submissions(config.loss_criteria, execute_team_name)

    for actor_email in holdout_execution_submissions.keys():
        # process list of submissions for the actor
        submissions = holdout_execution_submissions[actor_email]
        logging.info("Submitting {} submissions for actor email {}".format(len(submissions), actor_email))

        for submission in submissions:
            time_str = time_utils.convert_epoch_to_psudo_iso(submission.execution_epoch)

            existing_actor_submission_filepath = os.path.join(submission.global_submission_dirpath, submission.actor.name, time_str, submission.file.name)
            if not os.path.exists(existing_actor_submission_filepath):
                logging.error('Unable to find {}, cannot execute submission without container file.'.format(existing_actor_submission_filepath))
                continue

            # create a submissions directory to store a copy of the holdout containers. The containers run for holdout will be duplicated and stored in 2 places.
            holdout_container_submission_dir = os.path.join(config.submission_dir, submission.actor.name, time_str)
            holdout_actor_submission_filepath = os.path.join(holdout_container_submission_dir, submission.file.name)
            if not os.path.exists(holdout_container_submission_dir):
                logging.info('Creating directory to hold container image. {}'.format(holdout_container_submission_dir))
                os.makedirs(holdout_container_submission_dir)

            # Copy existing submission into holdout record
            logging.info('Copying container from {} to {}.'.format(existing_actor_submission_filepath, holdout_actor_submission_filepath))
            shutil.copyfile(existing_actor_submission_filepath, holdout_actor_submission_filepath)

            holdout_actor_results_dirpath = os.path.join(config.results_dir, submission.actor.name, time_str)
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

            cmd_str_list = ['sbatch', "--partition", v100_slurm_queue, "-n", "1", ":", "--partition", config.slurm_queue, "--gres=gpu:1", "-J", slurm_job_name, "--nice", "--parsable", "-o", slurm_output_filepath, config.slurm_script_file, submission.actor.name, holdout_actor_submission_filepath, holdout_actor_results_dirpath, config_filepath, submission.actor.email, slurm_output_filepath]
            logging.info('launching sbatch command: "{}"'.format(' '.join(cmd_str_list)))

            logging.info('Launching holdout computation for actor: {}'.format(submission.actor.name))
            logging.info('\tES Cross entropy loss: {}'.format(submission.cross_entropy))
            logging.info('\tSubmission container: {}'.format(holdout_actor_submission_filepath))
            logging.info('\tHoldout result dir: {}'.format(holdout_actor_results_dirpath))
            logging.info('\tConfig filepath: {}'.format(config_filepath))

            out = subprocess.Popen(cmd_str_list,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            stdout, stderr = out.communicate()

            # Check if there are no errors reported from sbatch
            if stderr == b'':
                job_id = int(stdout.strip())
                logging.info("Slurm job executed with job id: {}".format(job_id))
            else:
                logging.error("The slurm script: {} resulted in errors {}".format(config.slurm_script_file, stderr))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executes holdout data on actors that meet criteria")

    parser.add_argument('--config-file', type=str,
                        help='Filepath to the JSON file that describes the holdout execution',
                        default='config-holdout.json')

    parser.add_argument('--execute-team-name', type=str,
                        help='Executes the best model from team name. If None (or missing) all teams meeting the cross entropy requirement will be run.',
                        default=None)

    args = parser.parse_args()

    config_filepath = args.config_file
    config = Config.load_json(config_filepath)
    execute_team_name = args.execute_team_name

    handler = logging.handlers.RotatingFileHandler(config.log_file, maxBytes=100*1e6, backupCount=10) # 100MB
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[handler])

    logging.info('Starting parsing for holdout execution')
    logging.info(config)
    main(config_filepath, config, execute_team_name)
