#!/bin/bash

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


dir=$1
GPU_ID=$2
CONTAINER_NAME=$3
TASK_SCRIPT=$4


ACTIVE_DIR="/home/trojai/active_$GPU_ID"

CONTAINER_EXEC="/mnt/scratch/$CONTAINER_NAME"
RESULT_DIR=/mnt/scratch/results
SCRATCH_DIR="/mnt/scratch/container-scratch_$GPU_ID"

MODEL="$(basename $dir)"

export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU_ID

mkdir -p $RESULT_DIR
mkdir -p $SCRATCH_DIR

# create the active dir
mkdir -p $ACTIVE_DIR

# clean up scratch directory prior to running each model
rm -rf $SCRATCH_DIR/*
# pre-preemptively clean up the active directory
rm -rf $ACTIVE_DIR/*
# Copy model to the active folder to obscure its name
cp -r $dir/* $ACTIVE_DIR

echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] [evaluate_model.sh] Starting execution of $dir" >> "$RESULT_DIR/$CONTAINER_NAME.out" 2>&1
/usr/bin/time -f "execution_time %e" -o $RESULT_DIR/$MODEL-walltime.txt $TASK_SCRIPT $RESULT_DIR $MODEL $SCRATCH_DIR $CONTAINER_EXEC $ACTIVE_DIR $CONTAINER_NAME "${@:5}"
echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] [evaluate_model.sh] Finished executing $dir, returned status code: $?"

# copy result back to real output filename based on model name
if [[ -f $ACTIVE_DIR/result.txt ]]; then
	cp $ACTIVE_DIR/result.txt  $RESULT_DIR/$MODEL.txt
fi
