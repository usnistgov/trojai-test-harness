import logging
import logging.handlers
import fcntl
import os

from actor_executor.config import Config
from actor_executor import html_output
from actor_executor.actor import Actor, ActorManager
from actor_executor.submission import Submission, SubmissionManager
from actor_executor import time_utils

def main(config: Config, push_html: bool):
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
    html_output.update_html(config.html_repo_dir, actor_manager, submission_manager, config.execute_window, config.job_table_name, config.result_table_name, push_html, cur_epoch, config.accepting_submissions, config.slurm_queue)
    logging.debug('Finished updating website.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Applies updates to the front-end web-site.')
    parser.add_argument('--config-file', type=str,
                        help='The JSON file that describes all actors',
                        default='config.json')

    parser.add_argument("--no-push-html", dest='push_html',
                        help="Disables pushing the html web content to the Internet",
                        action='store_false')

    parser.set_defaults(push_html=True)
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
            main(config, args.push_html)
        except OSError as e:
            print('Server "{}", check-and-launch was already running when called.'.format(config.slurm_queue))
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)