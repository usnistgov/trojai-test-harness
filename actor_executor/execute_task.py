from actor_executor.trojai_config import TrojaiConfig
from actor_executor.leaderboard import Leaderboard
from actor_executor.mail_io import TrojaiMail
from actor_executor.drive_io import DriveIO
import logging
import traceback
import hashlib
import os


def compute_hash(filepath, buf_size=65536):
    sha256 = hashlib.sha256()
    pre, ext = os.path.splitext(filepath)
    output_filepath = pre + '.sha256'

    if not os.path.exists(output_filepath):
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(buf_size)
                if not data:
                    break
                sha256.update(data)

        with open(output_filepath, 'w') as f:
            f.write(sha256.hexdigest())

def main(trojai_config: TrojaiConfig, leaderboard: Leaderboard, data_split_name: str,
         vm_name: str, team_name: str, team_email: str, container_filepath: str, result_dirpath: str):

    logging.info('**************************************************')
    logging.info('Executing Container within VM for team: {} within VM: {}'.format(team_name, vm_name))
    logging.info('**************************************************')

    errors = ''
    submission_metadata_filepath = os.path.join(result_dirpath, team_name + '.metadata.json')
    error_filepath = os.path.join(result_dirpath, 'errors.txt')
    info_file = os.path.join(result_dirpath, 'info.json')

    try:
        vm_ip = trojai_config.vms[vm_name]
    except:
        msg = 'VM "{}" ended up in the wrong SLURM queue.\n{}'.format(vm_name, traceback.format_exc())
        errors += ":VM:"
        logging.error(msg)
        logging.error('config: "{}"'.format(trojai_config))
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" In Wrong SLURM Queue'.format(vm_name), message=msg)
        raise

    task = leaderboard.task

    # Step 1) Download the submission (if it does not exist)
    if not os.path.exists(container_filepath):
        logging.info('Downloading file for "{}" from "{}"'.format(team_name, team_email))
        submission_dir = os.path.dirname(container_filepath)
        submission_name = os.path.basename(container_filepath)

        g_drive = DriveIO(trojai_config.token_pickle_filepath)
        g_drive_file = g_drive.submission_download(team_email, submission_dir, submission_metadata_filepath)

        if submission_name != g_drive_file.name:
            logging.info('Name of file has changed since launching submission')
            submission_name = g_drive_file.name
            container_filepath = os.path.join(submission_dir, submission_name)

    # Step 2) Compute hash of container (if it does not exist)
    compute_hash(container_filepath)

    # Step 3) Run basic VM task checks: check_gpu
    errors += task.run_basic_checks(vm_ip, vm_name)

    # Step 4) Check task parameters in container (files and directories, schema checker)
    errors += task.run_container_checks(container_filepath)

    # Step 5) Run basic VM cleanups (scratch)
    errors += task.cleanup_vm(vm_ip, vm_name)

    # Step 6) Copy in and update permissions task data/scripts (submission, eval_scripts, training dataset, model dataset, other per-task data (tokenizers), source_data)

    # Step 7) Execute submission and check errors

    # Step 8) Copy out results

    # Step 9) Re-run basic VM cleanups
    errors += task.cleanup_vm(vm_ip, vm_name)

    # Step 10) Update info dictionary (execution, errors)

    pass

if __name__ == '__main__':
    import argparse

    # logs written to stdout are captured by slurm
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s")

    parser = argparse.ArgumentParser(description='Starts/Stops VMs')
    parser.add_argument('--team-name', type=str,
                        help='The team name',
                        required=True)
    parser.add_argument('--team-email', type=str,
                        help='The team email',
                        required=True)
    parser.add_argument('--container-filepath', type=str,
                        help='The filepath to download the container.',
                        required=True)
    parser.add_argument('--result-dirpath', type=str,
                        help='The result directory for the team',
                        required=True)
    parser.add_argument('--trojai-config-filepath', type=str,
                        help='The JSON file that describes the trojai round',
                        default='config.json')
    parser.add_argument('--leaderboard-name', type=str,
                        help='The name of the leaderboard')
    parser.add_argument('--data-split-name', type=str, help='The name of the data split that we are executing on.')
    parser.add_argument('--vm-name', type=str,
                        help='The name of the vm.',
                        required=True)

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    leaderboard = Leaderboard.load_json(trojai_config, args.leaderboard_name)

    main(trojai_config, leaderboard, args.data_split_name, args.vm_name, args.team_name, args.team_email, args.container_filepath, args.result_dirpath)



