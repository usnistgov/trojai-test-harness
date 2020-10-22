import os
import subprocess
import logging
import traceback

from drive_io import DriveIO
from config import Config
from mail_io import TrojaiMail


def check_gpu(host):
    child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'nvidia-smi'])
    return child.wait()


def copy_in_submission(host, submission_dir, submission_name):
    child = subprocess.Popen(['scp', '-q', submission_dir + "/" + submission_name, 'trojai@'+host+':\"/mnt/scratch/' + submission_name + '\"'])
    return child.wait()


def copy_in_eval_script(host, eval_script_path):
    child = subprocess.Popen(['scp', '-q', eval_script_path, 'trojai@'+host+':/home/trojai/evaluate_models.sh'])
    return child.wait()


def update_perms_eval_script(host):
    child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'chmod', 'u+rwx', '/home/trojai/evaluate_models.sh'])
    return child.wait()


def execute_submission(host, submission_name, queue_name, timeout='25h'):
    child = subprocess.Popen(['timeout', '-s', 'SIGKILL', timeout, 'ssh', '-q', 'trojai@'+host, '/home/trojai/evaluate_models.sh', submission_name, queue_name, '/home/trojai/data'])
    return child.wait()


def cleanup_scratch(host):
    child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'rm', '-rf', '/mnt/scratch/*'])
    return child.wait()


def copy_out_results(host, result_dir):
    child = subprocess.Popen(['scp', '-r', '-q', 'trojai@'+host+':/mnt/scratch/results/*', result_dir])
    return child.wait()


def write_errors(file, errors):
    with open(file, mode='w', encoding='utf-8') as f:
        f.write(errors)


if __name__ == "__main__":
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
    parser.add_argument('--submission-dir', type=str,
                        help='The submission dir for the team',
                        required=True)
    parser.add_argument('--result-dir', type=str,
                        help='The result dir for the team',
                        required=True)
    parser.add_argument('--config-file', type=str,
                        help='The JSON file that describes all actors',
                        default='config.json')
    parser.add_argument('--vm-name', type=str,
                        help='The name of the vm.',
                        required=True)

    args = parser.parse_args()

    team_name = args.team_name
    team_email = args.team_email
    submission_dir = args.submission_dir
    result_dir = args.result_dir
    config_file = args.config_file
    vm_name = args.vm_name

    logging.info('**************************************************')
    logging.info('Executing Container within VM for team: {} within VM: {}'.format(team_name, vm_name))
    logging.info('**************************************************')

    config = Config.load_json(config_file)
    sts = False
    if config.slurm_queue == 'sts':
        sts = True

    # where to serialize the GDriveFile object to json
    submission_metadata_file = os.path.join(result_dir, team_name + '.metadata.json')
    logging.info('Serializing executed file to "{}"'.format(submission_metadata_file))
    error_file = os.path.join(result_dir, 'errors.txt')

    errors = ""

    logging.info('Starting execution for "{}"'.format(team_name))

    logging.info('Downloading file for "{}" from "{}"'.format(team_name, team_email))

    g_drive = DriveIO(config.token_pickle_file)
    g_drive_file = g_drive.submission_download(team_email, submission_dir, submission_metadata_file, sts)
    submission_name = g_drive_file.name
    logging.info('Download complete for "{}".'.format(submission_name))

    try:
        vmIp = config.vms[vm_name]
    except:
        msg = 'VM "{}" ended up in the wrong SLURM queue.\n{}'.format(vm_name, traceback.format_exc())
        errors += ":VM:"
        logging.error(msg)
        logging.error('config: "{}"'.format(config))
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" In Wrong SLURM Queue'.format(vm_name), message=msg)
        raise

    logging.info('Checking GPU status')
    sc = check_gpu(vmIp)
    if sc != 0:
        msg = '"{}" GPU may be off-line with status code "{}".'.format(vm_name, sc)
        errors += ":GPU:"
        logging.error(msg)
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" GPU May be Offline'.format(vm_name), message=msg)

    logging.info('Performing Preventative Cleaning of the VM')
    sc = cleanup_scratch(vmIp)
    if sc != 0:
        logging.error(vm_name + ' Cleanup failed may have failed for VM "{}" with status code "{}"'.format(vm_name, sc))
        errors += ":Cleanup:"

    logging.info('Copying in "{}"'.format(submission_name))
    sc = copy_in_submission(vmIp, submission_dir, submission_name)
    if sc != 0:
        msg = '"{}" Submission copy in may have failed with status code "{}".'.format(vm_name, sc)
        logging.error(msg)
        errors += ":Copy in:"
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Copy In Failed'.format(vm_name), message=msg)

    logging.info('Copying in "{}"'.format(config.evaluate_script))
    sc = copy_in_eval_script(vmIp, config.evaluate_script)
    if sc != 0:
        msg = '"{}" Evaluate script copy in may have failed with status code "{}".'.format(vm_name, sc)
        logging.error(msg)
        errors += ":Copy in:"
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Holdout Copy In Failed'.format(vm_name), message=msg)

    logging.info('Updating eval permissions in "{}"'.format(config.evaluate_script))
    sc = update_perms_eval_script(vmIp)
    if sc != 0:
        msg = '"{}" Evaluate script update perms may have failed with status code "{}".'.format(vm_name, sc)
        logging.error(msg)
        errors += ":Copy in:"
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Holdout Copy In Failed'.format(vm_name), message=msg)

    logging.info('Starting Execution of ' + submission_name)
    if sts:
        executeStatus = execute_submission(vmIp, submission_name, config.slurm_queue, timeout="30m")
    else:
        executeStatus = execute_submission(vmIp, submission_name, config.slurm_queue, timeout="25h")

    logging.info("Execute status = " + str(executeStatus))
    if executeStatus == -9:
        logging.error('VM "{}" Execute submission "{}" timed out'.format(vm_name, submission_name))
        errors += ":Timeout:"
    elif executeStatus != 0:
        logging.error('VM "{}" execute submission "{}" may have failed.'.format(vm_name, submission_name))
        errors += ":Execute:"

    logging.info('Copying out results for ' + submission_name)
    sc = copy_out_results(vmIp, result_dir)
    if sc != 0:
        logging.error(vm_name + ' Copy out results may have failed for VM "{}" with status code "{}"'.format(vm_name, sc))
        errors += ":Copy out:"

    logging.info('Cleaning up for ' + submission_name)
    sc = cleanup_scratch(vmIp)
    if sc != 0:
        logging.error(vm_name + ' Cleanup failed may have failed for VM "{}" with status code "{}"'.format(vm_name, sc))
        errors += ":Cleanup:"

    logging.info('**************************************************')
    logging.info('Container Execution Complete for team: {}'.format(team_name))
    logging.info('**************************************************')

    if errors != "":
        write_errors(error_file, errors)
