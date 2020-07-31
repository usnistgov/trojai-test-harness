import os
import subprocess
import logging
import traceback

from config import Config, HoldoutConfig
from mail_io import TrojaiMail


def check_gpu(host):
    child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'nvidia-smi'])
    return child.wait()


def copy_in_submission(host, submission_dir, submission_name):
    child = subprocess.Popen(['scp', '-q', submission_dir + "/" + submission_name, 'trojai@'+host+':/mnt/scratch/' + submission_name])
    return child.wait()


def copy_in_models(host, model_dir):
    child = subprocess.Popen(['scp', '-q', '-r', model_dir, 'trojai@'+host+':/mnt/scratch/models'])
    return child.wait()


def copy_in_eval_script(host, eval_script_dir, eval_script_name):
    child = subprocess.Popen(['scp', '-q', eval_script_dir + "/" + eval_script_name, 'trojai@'+host+':/mnt/scratch/' + eval_script_name])
    return child.wait()


def update_perms_eval_script(host, eval_script_name):
    child = subprocess.Popen(['ssh', '-q', 'trojai@'+host, 'chmod', 'u+rwx', '/mnt/scratch/' + eval_script_name])
    return child.wait()


def execute_submission(host, eval_script_name, submission_name, queue_name, timeout='25h'):
    child = subprocess.Popen(['timeout', '-s', 'SIGKILL', timeout, 'ssh', '-q', 'trojai@'+host, '/mnt/scratch/' + eval_script_name, submission_name, queue_name, '/mnt/scratch/models'])
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
    parser.add_argument('--submission-filepath', type=str,
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
    parser.add_argument('--holdout-config-file', type=str,
                        help='The hold out configuration file',
                        required=True)

    args = parser.parse_args()

    team_name = args.team_name
    submission_filepath = args.submission_filepath
    result_dir = args.result_dir
    config_file = args.config_file
    vm_name = args.vm_name
    holdout_config_file = args.holdout_config_file

    logging.info('**************************************************')
    logging.info('Executing Holdout Data on Container within VM for team: {} within VM: {}'.format(team_name, vm_name))
    logging.info('**************************************************')

    config = Config.load_json(config_file)
    holdout_config = HoldoutConfig.load_json(holdout_config_file)

    # where to serialize the GDriveFile object to json
    submission_metadata_file = os.path.join(result_dir, team_name + '.metadata.json')
    logging.info('Serializing executed file to "{}"'.format(submission_metadata_file))
    error_file = os.path.join(result_dir, 'errors.txt')

    errors = ""

    logging.info('Starting execution for "{}"'.format(team_name))

    try:
        vmIp = config.vms[vm_name]
    except:
        msg = 'VM "{}" ended up in the wrong SLURM queue.\n{}'.format(vm_name, traceback.format_exc())
        errors += ":VM:"
        logging.error(msg)
        logging.error('config: "{}"'.format(config))
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Holdout In Wrong SLURM Queue'.format(vm_name), message=msg)
        raise

    logging.info('Checking GPU status')
    sc = check_gpu(vmIp)
    if sc != 0:
        msg = '"{}" GPU may be off-line with status code "{}".'.format(vm_name, sc)
        errors += ":GPU:"
        logging.error(msg)
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Holdout  GPU May be Offline'.format(vm_name), message=msg)

    logging.info('Performing Preventative Cleaning of the VM')
    sc = cleanup_scratch(vmIp)
    if sc != 0:
        logging.error(vm_name + ' Holdout cleanup failed may have failed for VM "{}" with status code "{}"'.format(vm_name, sc))
        errors += ":Cleanup:"

    logging.info('Copying in "{}"'.format(submission_filepath))
    submission_name = os.path.basename(submission_filepath)
    submission_dir = os.path.dirname(submission_filepath)
    sc = copy_in_submission(vmIp, submission_dir, submission_name)
    if sc != 0:
        msg = '"{}" Submission copy in may have failed with status code "{}".'.format(vm_name, sc)
        logging.error(msg)
        errors += ":Copy in:"
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Holdout Copy In Failed'.format(vm_name), message=msg)

    logging.info('Copying in "{}"'.format(holdout_config.evaluate_script))
    eval_script_name = os.path.basename(holdout_config.evaluate_script)
    eval_script_dir = os.path.dirname(holdout_config.evaluate_script)
    sc = copy_in_eval_script(vmIp, eval_script_dir, eval_script_name)
    if sc != 0:
        msg = '"{}" Evaluate script copy in may have failed with status code "{}".'.format(vm_name, sc)
        logging.error(msg)
        errors += ":Copy in:"
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Holdout Copy In Failed'.format(vm_name), message=msg)

    logging.info('Updating eval permissions in "{}"'.format(eval_script_name))
    sc = update_perms_eval_script(vmIp, eval_script_name)
    if sc != 0:
        msg = '"{}" Evaluate script update perms may have failed with status code "{}".'.format(vm_name, sc)
        logging.error(msg)
        errors += ":Copy in:"
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Holdout Copy In Failed'.format(vm_name), message=msg)

    logging.info('Copying in models: "{}"'.format(holdout_config.model_dir))
    sc = copy_in_models(vmIp, holdout_config.model_dir)
    if sc != 0:
        msg = '"{}" Model copy in may have failed with status code "{}."'.format(vm_name, sc)
        logging.error(msg)
        TrojaiMail().send(to='trojai@nist.gov', subject='VM "{}" Holdout Copy In Models Failed'.format(vm_name), message=msg)

    logging.info('Starting Holdout Execution of ' + submission_name)
    executeStatus = execute_submission(vmIp, eval_script_name, submission_name, config.slurm_queue, timeout="25h")

    logging.info("Execute status = " + str(executeStatus))
    if executeStatus == -9:
        logging.error('VM "{}" Holdout execute submission "{}" timed out'.format(vm_name, submission_name))
        errors += ":Timeout:"
    elif executeStatus != 0:
        logging.error('VM "{}" Holdout execute submission "{}" may have failed.'.format(vm_name, submission_name))
        errors += ":Execute:"

    logging.info('Copying out results for ' + submission_name)
    sc = copy_out_results(vmIp, result_dir)
    if sc != 0:
        logging.error(vm_name + ' Holdout copy out results may have failed for VM "{}" with status code "{}"'.format(vm_name, sc))
        errors += ":Copy out:"

    logging.info('Cleaning up for ' + submission_name)
    sc = cleanup_scratch(vmIp)
    if sc != 0:
        logging.error(vm_name + ' Holdout cleanup failed may have failed for VM "{}" with status code "{}"'.format(vm_name, sc))
        errors += ":Cleanup:"

    logging.info('**************************************************')
    logging.info('Holdout Container Execution Complete for team: {}'.format(team_name))
    logging.info('**************************************************')

    if errors != "":
        write_errors(error_file, errors)
