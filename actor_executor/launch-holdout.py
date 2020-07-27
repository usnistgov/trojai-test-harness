import logging
import logging.handlers

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
    holdout_execution_submissions = submission_manager.gather_Submissions(holdout_config.min_loss_criteria)

    for actor_email in holdout_execution_submissions.keys():
        print(actor_email)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executes holdout data on actors that meet criteria")

    parser.add_argument('--holdout-config-file', type=str,
                        help='The JSON file that describes the holdout execution',
                        default='holdout-config.json')

    args = parser.parse_args()

    holdout_config = HoldoutConfig.load_json(args.holdout_config_file)
    round_config = Config.load_json(holdoutConfig.round_config_filepath)

    handler = logging.handlers.RotatingFileHandler(holdoutConfig.log_file, maxBytes=100*1e6, backupCount=10) # 100MB
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[handler])

    logging.info('Starting parsing for holdout execution')
    main(round_config, holdout_config)
