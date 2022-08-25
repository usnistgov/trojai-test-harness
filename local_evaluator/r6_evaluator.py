import os
import subprocess
import time
import random
import shutil
import numpy as np
import sklearn.metrics
from collections import OrderedDict
import argparse
import json

from actor_executor import metrics
from actor_executor import time_utils


def eval(singularity_container, models_directory, example_subdirectory, scratch_directory, results_directory, disable_run_singularity_container, delete_scratch_directory, tokenizers_filepath, embeddings_filepath, gpu_ids=None):
    run_singularity_container = not disable_run_singularity_container
    delete_scratch_contents = delete_scratch_directory

    if gpu_ids is not None:
        cuda_devices = [int(item) for item in gpu_ids.split(',')]
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
                print(model_file_path)
                example_data_dirpath = os.path.join(model_dirpath, example_subdirectory)
                result_filepath = os.path.join(results_directory, model_dir_name + ".txt")

                # Make sure the model file and example data exists
                if not os.path.exists(model_file_path):
                    print("model_file_path={} missing".format(model_file_path))
                    continue
                if not os.path.exists(example_data_dirpath):
                    print("model_file_path={} missing".format(example_data_dirpath))
                    continue
                if not os.path.exists(os.path.join(model_dirpath, 'config.json')):
                    print("config_filepath={} missing".format(os.path.join(model_dirpath, 'config.json')))
                    continue

                with open(os.path.join(model_dirpath, 'config.json'), 'r') as fh:
                    model_config = json.load(fh)
                embedding_name = model_config['embedding_filename']

                emb_path = os.path.join(embeddings_filepath, embedding_name)
                tok_path = os.path.join(tokenizers_filepath, embedding_name)

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

                my_env['CUDA_VISIBLE_DEVICES'] = str(cuda_devices[device_index])
                # scratch_with_device = os.path.join(scratch_directory, str(cuda_devices[device_index]))
                scratch_with_device = scratch_directory
                print('Launching on GPU {} for model {}'.format(cuda_devices[device_index], model_dir_name))
                command = ['singularity', 'run', '--contain',
                           '-B', scratch_with_device,
                           '-B', model_dirpath,
                           '-B', emb_path,
                           '-B', tok_path,
                           '-B', results_directory, '--nv',
                           singularity_container,
                           '--model_filepath', model_file_path,
                           '--result_filepath', result_filepath,
                           '--scratch_dirpath', scratch_with_device,
                           '--tokenizer_filepath', tok_path,
                           '--embedding_filepath', emb_path,
                           '--examples_dirpath', example_data_dirpath]

                p = subprocess.Popen(command, env=my_env)
                p.wait()
                # processes[device_index] = p

                # if len(processes) == num_devices:
                #     # All GPUs are being used, we wait for one to become available
                #     all_processes_running = True
                #     while all_processes_running:
                #         for index in processes.keys():
                #             poll = processes[index].poll()
                #             # Process ended, that GPU is now available
                #             if poll is not None:
                #                 all_processes_running = False
                #                 device_index = index
                #                 break

                #         # No processes ended, wait a second before checking again
                #         if all_processes_running:
                #             time.sleep(1)
                #         # Found a GPU, delete the index from the processes
                #         else:
                #             del processes[device_index]
                # else:
                #     device_index = device_index + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Executes singularity container from TrojAI competition on a set of models to compute its cross entropy')
    #parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument('--singularity-container', type=str,
                        help='The singularity container to execute',
                        required=True)
    parser.add_argument('--model-directory', type=str,
                        help='Path to the directory that contains all models to execute',
                        required=True)
    parser.add_argument('--example-subdirectory', type=str,
                        help='Name of the subdirectory containing the example data',
                        default='example_data')
    parser.add_argument('--scratch-directory', type=str,
                        help='Path to the scratch directory',
                        default='./scratch')
    parser.add_argument('--results-directory', type=str,
                        help='Path to the results directory',
                        default='./results')
    parser.add_argument('--disable-run-singularity-container',
                        help='Disables running the singularity container and only processes the results.',
                        action='store_true', default=False)
    parser.add_argument('--team-name', help='Optional team name to display what team was executed',
                    default='Not specified')
    parser.add_argument('--tokenizers-directory', type=str,
                        help='Path to the tokenizers',
                        required=True)
    parser.add_argument('--embeddings-directory', type=str,
                        help='Path to the tokenizers',
                        required=True)
    parser.add_argument('--delete-scratch-directory', help='Enables deleting scratch directory',
                        action='store_true', default=False)
    parser.add_argument('--gpu-ids', help='delimited list of gpu ids', type=str, default=None)

    # add team name and container name

    args = parser.parse_args()

    # Load parameters
    singularity_container = args.singularity_container
    models_directory = args.model_directory
    example_subdirectory = args.example_subdirectory
    scratch_directory = args.scratch_directory
    results_directory = args.results_directory
    tokenizers_filepath = args.tokenizers_directory
    embeddings_filepath = args.embeddings_directory
    current_epoch = time_utils.get_current_epoch()
    team_name = args.team_name

    print(args)

    eval(singularity_container, models_directory, example_subdirectory, scratch_directory, results_directory, args.disable_run_singularity_container, args.delete_scratch_directory, tokenizers_filepath, embeddings_filepath, args.gpu_ids)