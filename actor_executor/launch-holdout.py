import os
import logging
import logging.handlers

import time_utils
from config import Config
from holdout_config import HoldoutConfig
from submission import Submission, SubmissionManager


def main(round_config: Config, holdout_config: HoldoutConfig) -> None:
    submission_manager = SubmissionManager.load_json(round_config.submissions_json_file)
    logging.debug('Loaded submission_manager from filepath: {}'.format(round_config.submissions_json_file))
    logging.debug(submission_manager)

    # Gather submissions based on criteria
    min_loss_criteria = holdout_config.min_loss_criteria

    # Key = actor, value = submission that is best that meets critera
    holdout_execution_submissions = submission_manager.gather_submissions(holdout_config.min_loss_criteria)

    for actor_email in holdout_execution_submissions.keys():
        submission = holdout_execution_submissions[actor_email]
        time_str = time_utils.convert_epoch_to_psudo_iso(submission.execution_epoch)
        actor_submission_filepath = os.path.join(submission.global_submission_dirpath, submission.actor.name, time_str, submission.file.name)
        print(actor_submission_filepath)

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
    main(round_config, holdout_config)
