# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import logging.handlers
import fcntl
import os

from actor_executor.config import Config
from actor_executor import html_output
from actor_executor.actor import ActorManager
from actor_executor.submission import SubmissionManager
from actor_executor import time_utils


def main(config: Config, commit_and_push: bool):
    cur_epoch = time_utils.get_current_epoch()

    # load the instance of ActorManager from the serialized json file
    actor_manager = ActorManager.load_json(config.actor_json_file)
    logging.debug('Loaded actor_manager from filepath: {}'.format(config.actor_json_file))
    logging.debug(actor_manager)

    # load the instance of SubmissionManager from the serialized json file
    submission_manager = SubmissionManager.load_json(config.submissions_json_file)
    logging.debug('Loaded submission_manager from filepath: {}'.format(config.submissions_json_file))
    logging.debug(submission_manager)

    # Update web-site
    logging.debug('Updating website.')
    html_output.update_html(config.html_repo_dir, actor_manager, submission_manager, config.execute_window, config.job_table_name, config.result_table_name, commit_and_push, cur_epoch, config.accepting_submissions, config.slurm_queue)
    logging.debug('Finished updating website.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Applies updates to the leaderboard web-site.')
    parser.add_argument('--config-file', type=str,
                        help='The JSON file that describes all actors',
                        default='config.json')

    parser.add_argument("--commit-and-push", dest='commit_and_push',
                        help="Enables pushing the html web content to the Internet",
                        action='store_true')

    parser.set_defaults(commit_and_push=True)
    args = parser.parse_args()

    config = Config.load_json(args.config_file)

    # PidFile ensures that this script is only running once
    print('Attempting to acquire PID file lock.')
    lock_file = '/var/lock/trojai-{}-lockfile'.format(config.slurm_queue)

    with open(lock_file, 'w') as f:
        try:
            fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print('  PID lock acquired')
            # make sure intermediate folders to the logfile exists
            parent_fp = os.path.dirname(config.log_file)
            if not os.path.exists(parent_fp):
                os.makedirs(parent_fp)
            # Add the log message handler to the logger
            handler = logging.handlers.RotatingFileHandler(config.log_file, maxBytes=100*1e6, backupCount=10) # 100MB
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                                handlers=[handler])

            logging.debug('PID file lock acquired in directory {}'.format(config.submission_dir))
            main(config, args.commit_and_push)
        except OSError as e:
            print('Server "{}", check-and-launch was already running when called.'.format(config.slurm_queue))
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)