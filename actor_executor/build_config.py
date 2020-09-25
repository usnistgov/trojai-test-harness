# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import argparse

from actor_executor import time_utils
from actor_executor.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct a config.json file.')
    parser.add_argument('--actor-json-file', type=str,
                        help='The JSON file that describes all actors ',
                        required=True)

    parser.add_argument('--submissions-json-file', type=str,
                        help='The JSON file that describes all actors ',
                        required=True)

    parser.add_argument('--submission-dir', type=str,
                        help='Location of submissions',
                        required=True)

    parser.add_argument('--log-file', type=str,
                        help='Log file for check and launch',
                        required=True)

    parser.add_argument('--execute-window', type=time_utils.parse_time,
                        help='Timeout time for how long execution must wait until next execution, i.e. "7d" for 7 days',
                        default="7d")

    parser.add_argument("--ground-truth-dir", type=str,
                        help="The ground truth directory",
                        required=True)

    parser.add_argument("--html-repo-dir", type=str,
                        help="The directory where web-site repo is stored.",
                        required=True)

    parser.add_argument("--results-dir", type=str,
                        help="The directory where results will be placed",
                        required=True)

    parser.add_argument("--models-dir", type=str,
                        help="The directory where results will be placed",
                        required=True)

    parser.add_argument('--token-pickle-file', type=str,
                        help='Path token.pickle file holding the oauth keys. If token.pickle is missing, but credentials have been provided, token.pickle will be generated after opening a web-browser to have the user accept the app permissions',
                        default='token.pickle')

    parser.add_argument('--slurm-script', type=str,
                        help='The slurm script that each actor executes',
                        required=True)

    parser.add_argument('--evaluate-script', type=str,
                        help='The evaluation script that executes on the VM',
                        default='/mnt/isgnas/project/ai/trojai/trojai-test-harness/vm_scripts/evaluate_models.sh'
                        )

    parser.add_argument('--sts', action='store_true')
    parser.add_argument('--accepting-submissions', action='store_true')

    parser.add_argument('--output-filepath', type=str,
                        help='where to place the resulting config file',
                        default='config.json')

    args = parser.parse_args()

    # with model copy in, the VMs do not need to be split between STS and ES
    vms = dict()
    vms['gpu-vm-61'] = '192.168.200.4'
    vms['gpu-vm-db'] = '192.168.200.7'
    vms['gpu-vm-3b'] = '192.168.200.2'
    vms['gpu-vm-60'] = '192.168.200.3'
    vms['gpu-vm-86'] = '192.168.200.5'
    vms['gpu-vm-da'] = '192.168.200.6'

    if args.sts:
        results_table_name = 'test-results'
        jobs_table_name = 'test-jobs'
        slurm_queue = 'sts'
    else:
        results_table_name = 'results'
        jobs_table_name = 'jobs'
        slurm_queue = 'es'

    MB_limit = 1
    log_file_byte_limit = int(MB_limit * 1024 * 1024)

    config = Config(args.actor_json_file, args.submissions_json_file, args.log_file, args.submission_dir, args.execute_window, args.ground_truth_dir, args.html_repo_dir, args.models_dir, args.results_dir, args.token_pickle_file, args.slurm_script, jobs_table_name, results_table_name, vms, slurm_queue, args.evaluate_script)

    ofp = args.output_filepath
    if not ofp.endswith('.json'):
        raise IOError('Invalid output filename, must end with ".json"')
    config.save_json(ofp)
