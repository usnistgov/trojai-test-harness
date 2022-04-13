#!/bin/bash

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


CONTAINER_NAME=$1
QUEUE_NAME=$2
TIMEOUT=$3

MODEL_DIR=/home/trojai/models

CHILD_JOB_NAME="child_$SLURM_JOB_ID"

# find all the 'id-' model files and shuffle their iteration order
for dir in `find $MODEL_DIR -maxdepth 1 -type d | shuf`
do
	# check that the directory is not the root MODEL_DIR
	if [ "$dir" != "$MODEL_DIR" ]; then
		# check that the directory starts with "id"
		MODEL="$(basename $dir)"

		if [[ $MODEL == id* ]] ; then
			# launch the job
			sbatch --cpus-per-task=30 --gres=gpu:1 --job-name=$CHILD_JOB_NAME --partition=$QUEUE_NAME ./evaluate_model.sh $CONTAINER_NAME $QUEUE_NAME $dir
		fi
	fi
done

# wait until timeout is reached or all jobs are finished
while :
do
  # timeout was reached
  if [ $SECONDS -ge $TIMEOUT ] ; then
    scancel --jobname=$CHILD_JOB_NAME
    exit 9
  fi

  # all jobs finished
  if [ `squeue --name=$CHILD_JOB_NAME --noheader | wc -l` = 0 ] ; then
    exit 0
  fi

  sleep 1m
done