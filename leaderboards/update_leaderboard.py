# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import logging.handlers
import fcntl
import os

from leaderboards.trojai_config import TrojaiConfig
from leaderboards import html_output
from leaderboards.actor import ActorManager
from leaderboards.submission import SubmissionManager
from leaderboards.leaderboard import Leaderboard


def main(trojai_config: TrojaiConfig, commit_and_push: bool):
    active_leaderboards = {}
    active_submission_managers = {}
    archive_leaderboards = {}

    for leaderboard_name in trojai_config.active_leaderboard_names:
        leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
        active_leaderboards[leaderboard_name] = leaderboard
        submission_manager = SubmissionManager.load_json(leaderboard.submissions_filepath, leaderboard.name)
        active_submission_managers[leaderboard_name] = submission_manager

    for leaderboard_name in trojai_config.archive_leaderboard_names:
        leaderboard = Leaderboard.load_json(trojai_config, leaderboard_name)
        archive_leaderboards[leaderboard_name] = leaderboard

    # load the instance of ActorManager from the serialized json file
    actor_manager = ActorManager.load_json(trojai_config)
    logging.debug('Loaded actor_manager from filepath: {}'.format(trojai_config.actors_filepath))
    logging.debug(actor_manager)

    # Update web-site
    logging.debug('Updating website.')
    html_output.update_html_pages(trojai_config, actor_manager, active_leaderboards, active_submission_managers, archive_leaderboards, commit_and_push=commit_and_push)
    logging.debug('Finished updating website.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Applies updates to the leaderboards web-site.')
    parser.add_argument('--trojai-config-file', type=str,
                        help='The JSON file that describes trojai', required=True)

    parser.add_argument("--commit-and-push", dest='commit_and_push',
                        default=False,
                        help="Enables pushing the html web content to the Internet",
                        action='store_true')

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_file)

    # PidFile ensures that this script is only running once
    print('Attempting to acquire PID file lock.')
    lock_file = '/var/lock/trojai-lockfile'

    with open(lock_file, 'w') as f:
        try:
            fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print('  PID lock acquired')
            # make sure intermediate folders to the logfile exists
            parent_fp = os.path.dirname(trojai_config.log_filepath)
            if not os.path.exists(parent_fp):
                os.makedirs(parent_fp)
            # Add the log message handler to the logger
            handler = logging.handlers.RotatingFileHandler(trojai_config.log_filepath, maxBytes=100*1e6, backupCount=10) # 100MB
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                                handlers=[handler])

            logging.debug('PID file lock acquired')
            main(trojai_config, args.commit_and_push)
        except OSError as e:
            print('TrojAI check-and-launch was already running when called.')
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)