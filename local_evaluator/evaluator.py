import os
import subprocess
import time
import random
import shutil
import numpy as np
import sklearn.metrics
from collections import OrderedDict
import configargparse

from actor_executor import metrics
from actor_executor import time_utils

parser = configargparse.ArgParser(default_config_files=['./config.txt'], description='Executs singularity container from TrojAI competition on a set of models '
                                             'to compute its cross entropy')
parser.add_argument('--config', is_config_file=True, help='Config file path')
parser.add_argument('--team-name', help='Optional team name to display what team was executed',
                    default='Not specified')
parser.add_argument('--ground-truth-directory', type=str,
                    help='Path to ground truth (if not set then will use models_directory)')
parser.add_argument('--singularity-container', type=str,
                    help='The singularity container to execute',
                    required=True)
parser.add_argument('--output-filepath', type=str,
                    help='The output file for outputting CSV results',
                    default='./output.csv')
parser.add_argument('--model-directory', type=str,
                    help='Path to the directory that contains all models to execute',
                    required=True)
parser.add_argument('--scratch-directory', type=str,
                    help='Path to the scratch directory',
                    default='./scratch')
parser.add_argument('--results-directory', type=str,
                    help='Path to the results directory',
                    default='./results')
parser.add_argument('--disable-run-singularity-container',
                    help='Disables running the singularity container and only processes the results.',
                    action='store_true', default=False)
parser.add_argument('--delete-scratch-directory', help='Enables deleting scratch directory',
                    action='store_true', default=False)
parser.add_argument('--gpu-ids', help='delimited list of gpu ids', type=str, default=None)

# add team name and container name

args = parser.parse_args()

# Load parameters
singularity_container = args.singularity_container
models_directory = args.model_directory
scratch_directory = args.scratch_directory
results_directory = args.results_directory
output_filepath = args.output_filepath
current_epoch = time_utils.get_current_epoch()
team_name = args.team_name

if args.ground_truth_directory is not None:
    ground_truth_directory = args.ground_truth_directory
else:
    ground_truth_directory = models_directory

if not os.path.exists(os.path.dirname(output_filepath)):
    os.makedirs(os.path.dirname(output_filepath))

run_singularity_container = not args.disable_run_singularity_container
delete_scratch_contents = args.delete_scratch_directory

if args.gpu_ids is not None:
    cuda_devices = [int(item) for item in args.gpu_ids.split(',')]
else:
    cuda_devices = []

if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Launch singularity container on all models in directory. this will execute 1 process per GPU and round-robin
my_env = os.environ.copy()

if run_singularity_container:
    if len(cuda_devices) == 0:
        if 'CUDA_VISIBLE_DEVICES' in my_env:
            cuda_devices = my_env['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            import pycuda
            import pycuda.driver as drv
            drv.init()
            cuda_devices = [*range(drv.Device.count())]

    print("Using {} GPU ids.".format(cuda_devices))

    # Create a separate scratch directory for each CUDA device
    for device_id in cuda_devices:
        scratch_with_device = os.path.join(scratch_directory, str(device_id))
        if not os.path.exists(scratch_with_device):
            os.makedirs(scratch_with_device)

    device_index = 0
    num_devices = len(cuda_devices)
    processes = {}

    # Main loop over all models
    for model_dir_name in os.listdir(models_directory):
        model_dirpath = os.path.join(models_directory, model_dir_name)

        if os.path.isdir(model_dirpath):
            model_file_path = os.path.join(model_dirpath, 'model.pt')
            example_data_dirpath = os.path.join(model_dirpath, 'example_data')
            result_filepath = os.path.join(results_directory, model_dir_name + ".txt")

            # Make sure the model file and example data exists
            if not os.path.exists(model_file_path) or not os.path.exists(example_data_dirpath):
                continue

            my_env['CUDA_VISIBLE_DEVICES'] = str(cuda_devices[device_index])
            scratch_with_device = os.path.join(scratch_directory, str(cuda_devices[device_index]))
            print('Launching on GPU {} for model {}'.format(cuda_devices[device_index], model_dir_name))
            command = ['singularity', 'run', '--contain',
                       '-B', scratch_with_device,
                       '-B', model_dirpath,
                       '-B', results_directory, '--nv',
                       singularity_container,
                       '--model_filepath', model_file_path,
                       '--result_filepath', result_filepath,
                       '--scratch_dirpath', scratch_with_device,
                       '--examples_dirpath', example_data_dirpath]

            p = subprocess.Popen(command, env=my_env)
            processes[device_index] = p

            if len(processes) == num_devices:
                # All GPUs are being used, we wait for one to become available
                all_processes_running = True
                while all_processes_running:
                    for index in processes.keys():
                        poll = processes[index].poll()
                        # Process ended, that GPU is now available
                        if poll is not None:
                            all_processes_running = False
                            device_index = index
                            break

                    # No processes ended, wait a second before checking again
                    if all_processes_running:
                        time.sleep(1)
                    # Found a GPU, delete the index from the processes
                    else:
                        del processes[device_index]
            else:
                device_index = device_index + 1

    # Clean up scratch directory if requested
    if delete_scratch_contents:
        for filename in os.listdir(scratch_directory):
            file_path = os.path.join(scratch_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

# Process results
if os.path.isdir(results_directory):
    predictions_list = list()
    targets_list = list()

    for result_filename in os.listdir(results_directory):
        if result_filename.endswith('.txt'):
            model_name = result_filename.split('.')[0]
            result_filepath = os.path.join(results_directory, result_filename)
            ground_truth_filepath = os.path.join(ground_truth_directory, model_name, 'ground_truth.csv')
            if os.path.exists(ground_truth_filepath) and os.path.exists(result_filepath):
                with open(ground_truth_filepath) as truth_file:
                    file_contents = truth_file.readline().strip()
                    ground_truth = float(file_contents)
                    targets_list.append(ground_truth)

                with open(result_filepath) as result_file:
                    file_contents = result_file.readline().strip()
                    result = float(file_contents)
                    predictions_list.append(result)

    predictions = np.array(predictions_list).reshape(-1, 1)
    targets = np.array(targets_list).reshape(-1, 1)

    # if prediction is nan, then replace with guess (0.5)
    predictions[np.isnan(predictions)] = 0.5

    elementwise_cross_entropy = metrics.elementwise_binary_cross_entropy(predictions, targets)
    ce = float(np.mean(elementwise_cross_entropy))
    ce_95_ci = metrics.cross_entropy_confidence_interval(elementwise_cross_entropy)
    brier_score = metrics.binary_brier_score(predictions, targets)

    time_str = time_utils.convert_epoch_to_psudo_iso(current_epoch)

    header = 'team name, timestamp, singularity container, models directory, ground truth directory, cross entroy, cross entropy 95% CI, brier score\n'
    write_header = False

    if not os.path.exists(output_filepath):
        write_header = True

    with open(output_filepath, "a") as output_file:
        if write_header:
            output_file.write(header)

        new_line = "{}, {}, {}, {}, {}, {}, {}, {}\n".format(team_name, time_str, singularity_container, models_directory, ground_truth_directory, ce, ce_95_ci, brier_score)
        output_file.write(new_line)
