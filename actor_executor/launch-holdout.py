import logging
from config import Config
from holdout_config import HoldoutConfig
from submission import Submission, SubmissionManager


def main(config: Config, holdoutConfig: HoldoutConfig) -> None:
    submission_manager = SubmissionManager.load_json(config.submissions_json_file)
    logging.debug('Loaded submission_manager from filepath: {}'.format(config.submissions_json_file))
    logging.debug(submission_manager)

    # Gather submissions based on criteria
    min_loss_criteria = holdoutConfig.min_loss_criteria

    # Key = actor, value = submission that is best that meets critera
    holdout_execution_submissions = dict()

    for actor in submission_manager.__submissions.keys():
        submissions = submission_manager[actor]
        best_loss = 42.0
        best_submission = None

        for submission in submissions:
            if submission.score < best_loss:
                best_submission = submission
                best_loss = submission.score

        if best_loss < min_loss_criteria:
            holdout_execution_submissions[actor] = best_submission

    for actor in holdout_execution_submissions.keys():
        print(actor)







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executes holdout data on actors that meet criteria")

    parser.add_argument('--holdout-config-file', type=str,
                        help='The JSON file that describes the holdout execution',
                        default='holdout-config.json')

    args = parser.parse_args()

    holdoutConfig = HoldoutConfig.load_json(args.holdout_config_file)
    roundConfig = Config.load_json(holdoutConfig.round_config_file)

    handler = logging.handlers.RotatingFileHandler(holdoutConfig.log_file, maxBytes=100*1e6, backupCount=10) # 100MB
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[handler])

    logging.log('Starting parsing for holdout execution')
