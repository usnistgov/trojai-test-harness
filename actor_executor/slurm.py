import logging
import subprocess
import traceback


def squeue(job_name: str, queue_name: str):
    out = subprocess.Popen(['squeue', "-n", str(job_name), "-p", str(queue_name), "-o", "%T"],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()

    if stderr != b'':
        # TODO figure what went wrong with slurm (email dev team)
        logging.error("Slurm is no longer online, error = {}".format(stderr))
        logging.error(traceback.format_exc())
        raise RuntimeError("Slurm is no longer online, error = {}".format(stderr))
    return stdout, stderr


def sinfo_node_query(queue_name: str, state: str):
    out = subprocess.Popen(['sinfo', '-t', state, '-p', queue_name, '-o', '%D', '-h'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()

    if stderr != b'':
        logging.error("Slurm is no longer online, error = {}".format(stderr))
        return '0'

    if stdout == b'':
        return '0'

    return stdout.decode('utf-8').strip()
