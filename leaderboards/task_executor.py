from leaderboards.trojai_config import TrojaiConfig
from leaderboards.leaderboard import Leaderboard
from leaderboards.mail_io import TrojaiMail
from leaderboards.drive_io import DriveIO
from leaderboards import json_io
from leaderboards import hash_utils
import time
import logging
import traceback
import os



def main(trojai_config: TrojaiConfig, leaderboard: Leaderboard, data_split_name: str,
         vm_name: str, team_name: str, team_email: str, submission_filepath: str, result_dirpath: str, custom_remote_home=None, custom_remote_scratch=None, job_id=None,
         custom_metaparameter_filepath=None, subset_model_ids=None):

    logging.info('**************************************************')
    logging.info('Executing Container within VM for team: {} within VM: {}'.format(team_name, vm_name))
    logging.info('**************************************************')

    errors = ''
    info_dict = {}
    submission_dirpath = os.path.dirname(submission_filepath)

    if not os.path.exists(submission_dirpath):
        os.makedirs(submission_dirpath)

    if not os.path.exists(result_dirpath):
        os.makedirs(result_dirpath)

    submission_metadata_filepath = os.path.join(submission_dirpath, team_name + '.metadata.json')
    # error_filepath = os.path.join(result_dirpath, 'errors.txt')
    info_file = os.path.join(result_dirpath, 'info.json')

    try:
        if vm_name == 'local':
            vm_ip = 'local'
        else:
            vm_ip = trojai_config.vms[vm_name]
    except:
        msg = 'VM "{}" ended up in the wrong SLURM queue.\n{}'.format(vm_name, traceback.format_exc())
        errors += ":VM:"
        logging.error(msg)
        logging.error('config: "{}"'.format(trojai_config))
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" In Wrong SLURM Queue'.format(vm_name), message=msg)
        raise


    custom_remote_scratch_with_job_id = None

    if vm_ip == 'local':
        custom_remote_scratch_with_job_id = os.path.join(custom_remote_scratch, job_id)

    task = leaderboard.task
    dataset = leaderboard.get_dataset(data_split_name)
    train_dataset = leaderboard.get_dataset(Leaderboard.TRAIN_DATASET_NAME)

    # Step 1) Download the submission (if it does not exist)
    if not os.path.exists(submission_filepath):
        logging.info('Downloading file for "{}" from "{}"'.format(team_name, team_email))
        submission_dir = os.path.dirname(submission_filepath)
        submission_name = os.path.basename(submission_filepath)

        g_drive = DriveIO(trojai_config.token_pickle_filepath)
        g_drive_file = g_drive.submission_download(team_email, submission_dir, submission_metadata_filepath, leaderboard.name, data_split_name)

        if submission_name != g_drive_file.name:
            logging.info('Name of file has changed since launching submission')
            submission_name = g_drive_file.name
            submission_filepath = os.path.join(submission_dir, submission_name)

    # Step 2) Compute hash of container (if it does not exist)
    hash_utils.compute_hash(submission_filepath)

    # Step 3) Run basic VM task checks: check_gpu
    errors += task.run_basic_checks(vm_ip, vm_name)

    # Step 4) Check task parameters in container (files and directories, schema checker)
    errors += task.run_submission_checks(submission_filepath)

    # Step 5) Run basic VM cleanups (scratch)
    errors += task.cleanup_vm(vm_ip, vm_name, custom_remote_home, custom_remote_scratch_with_job_id)

    # Add some delays
    time.sleep(2)

    # Step 6) Copy in and update permissions task data/scripts (submission, eval_scripts, training dataset, model dataset, other per-task data (tokenizers), source_data)
    errors += task.copy_in_task_data(vm_ip, vm_name, submission_filepath, dataset, train_dataset, custom_remote_home, custom_remote_scratch_with_job_id, custom_metaparameter_filepath)

    # Add some delays
    time.sleep(2)

    # Step 7) Execute submission and check errors
    errors += task.execute_submission(vm_ip, vm_name, submission_filepath, dataset, train_dataset, info_dict, custom_remote_home, custom_remote_scratch_with_job_id, custom_metaparameter_filepath, subset_model_ids)

    # Add some delays
    time.sleep(2)

    # Step 8) Copy out results
    errors += task.copy_out_results(vm_ip, vm_name, result_dirpath, custom_remote_home, custom_remote_scratch_with_job_id)

    # Add some delays
    time.sleep(2)

    # Step 9) Re-run basic VM cleanups
    errors += task.cleanup_vm(vm_ip, vm_name, custom_remote_home, custom_remote_scratch_with_job_id)

    logging.info('**************************************************')
    logging.info('Container Execution Complete for team: {}'.format(team_name))
    logging.info('**************************************************')

    # Step 10) Update info dictionary (execution, errors)
    info_dict['errors'] = errors
    task.package_results(result_dirpath, info_dict)

    # delete job ID scratch
    if vm_ip == 'local':
        if os.path.exists(custom_remote_scratch_with_job_id):
            os.rmdir(custom_remote_scratch_with_job_id)


    # Build per model execution time dictionary
    model_execution_time_dict = dict()
    for model_execution_time_file_name in os.listdir(result_dirpath):
        if not model_execution_time_file_name.endswith('-walltime.txt'):
            continue

        model_name = model_execution_time_file_name.split('-walltime')[0]
        model_execution_time_filepath = os.path.join(result_dirpath, model_execution_time_file_name)

        if not os.path.exists(model_execution_time_filepath):
            continue

        try:
            with open(model_execution_time_filepath, 'r') as execution_time_fh:
                line = execution_time_fh.readline().strip()
                while line:
                    if line.startswith('execution_time'):
                        toks = line.split(' ')
                        model_execution_time_dict[model_name] = float(toks[1])
                    line = execution_time_fh.readline().strip()

        except:
            pass  # Do nothing if file fails to parse
        # delete the walltime file to avoid cluttering the output folder
        os.remove(model_execution_time_filepath)

    info_dict['model_execution_runtimes'] = model_execution_time_dict

    json_io.write(info_file, info_dict)


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
                        help='The name of the leaderboards')
    parser.add_argument('--data-split-name', type=str, help='The name of the data split that we are executing on.')
    parser.add_argument('--vm-name', type=str,
                        help='The name of the vm.',
                        required=True)
    parser.add_argument('--custom-remote-home', type=str,
                        help='The custom home directory to stage scripts in',
                        default=None)
    parser.add_argument('--custom-remote-scratch', type=str,
                        help='The custom scratch directory',
                        default=None)

    parser.add_argument('--custom-metaparameters-filepath', type=str, help='The custom metaparameters filepath to use', default=None)
    parser.add_argument('--model-ids-subset', nargs='*', help='The list of model IDs to subset when launching', default=None)

    parser.add_argument('--job-id', type=str, help='The slurm job ID', default=None)

    args = parser.parse_args()

    trojai_config = TrojaiConfig.load_json(args.trojai_config_filepath)
    leaderboard = Leaderboard.load_json(trojai_config, args.leaderboard_name)

    main(trojai_config, leaderboard, args.data_split_name, args.vm_name, args.team_name, args.team_email, args.container_filepath, args.result_dirpath, args.custom_remote_home, args.custom_remote_scratch, args.job_id, args.custom_metaparameters_filepath, args.model_ids_subset)



