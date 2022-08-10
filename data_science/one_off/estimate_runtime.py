# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import numpy as np
from matplotlib import pyplot as plt

from leaderboards import time_utils


def find_dirs(fp):
    # find all directories in the results folder
    dirs = [d for d in os.listdir(fp) if os.path.isdir(os.path.join(fp, d))]
    return dirs


def main(test_harness_dirpath, server, output_dirpath):

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    output_filepath = os.path.join(output_dirpath, 'estimated-runtimes.csv')
    with open(output_filepath, 'w') as output_fh:
        output_fh.write('team_name,execution_time_stamp,runtime_seconds\n')

        results_fp = os.path.join(test_harness_dirpath, server, 'results')

        runtimes = list()

        # find all team directories in the results folder
        teams = find_dirs(results_fp)
        teams.sort()

        for team in teams:
            team_fp = os.path.join(results_fp, team)
            # find all executions
            runs = find_dirs(team_fp)
            runs.sort()

            for run in runs:
                run_fp = os.path.join(team_fp, run)

                log_file_name = '{}.{}.log.txt'.format(team, server)
                log_file_path = os.path.join(run_fp, log_file_name)

                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r') as fh:
                        line = fh.readline()
                        start_time_str = line[0:19]
                        while line:
                            end_time_str = line[0:19]
                            line = fh.readline()

                    start_time_str = start_time_str.replace(' ', 'T')
                    end_time_str = end_time_str.replace(' ', 'T')
                    start_epoch = time_utils.convert_to_epoch(start_time_str)
                    end_epoch = time_utils.convert_to_epoch(end_time_str)
                    runtime_seconds = end_epoch - start_epoch
                    runtimes.append(runtime_seconds)
                    output_fh.write('{},{},{}\n'.format(team, run, runtime_seconds))

    runtimes = np.asarray(runtimes)

    fig = plt.figure(figsize=(5, 4), dpi=100)
    plt.hist(runtimes, bins=100)
    plt.title('Container Runtimes (s)')
    plt.savefig(os.path.join(output_dirpath, 'runtimes-histogram.png'))
    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to estimate job runtime from the log file timestamps.')
    parser.add_argument('--test-harness-dirpath', type=str, required=True)
    parser.add_argument('--server', type=str, required=True)
    parser.add_argument('--output-dirpath', type=str, required=True)

    args = parser.parse_args()

    main(args.test_harness_dirpath, args.server, args.output_dirpath)
