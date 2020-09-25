#!/bin/bash

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


CONTAINER_NAME=$1
QUEUE_NAME=$2

MODEL_DIR=/home/trojai/data
ACTIVE_DIR=/home/trojai/active

CONTAINER_EXEC=/mnt/scratch/$CONTAINER_NAME
RESULT_DIR=/mnt/scratch/results
SCRATCH_DIR=/mnt/scratch/scratch

mkdir -p $RESULT_DIR
mkdir -p $SCRATCH_DIR

# only use the name obfuscation outside the STS queue
if [[ "$QUEUE_NAME" != "sts" ]]; then
	mkdir -p $ACTIVE_DIR
fi


# find all the 'id-' model files and shuffle their iteration order
for dir in `find $MODEL_DIR -maxdepth 1 -type d | shuf`
do
	# check that the directory is not the root MODEL_DIR
	if [ "$dir" != "$MODEL_DIR" ]; then
		# check that the directory starts with "id"
		MODEL="$(basename $dir)"

		if [[ $MODEL == id* ]] ; then

			if [[ "$QUEUE_NAME" == "sts" ]]; then
				singularity run --contain -B /mnt/scratch -B /home/trojai/data --nv $CONTAINER_EXEC --model_filepath $dir/model.pt --result_filepath $RESULT_DIR/$MODEL.txt --scratch_dirpath $SCRATCH_DIR --examples_dirpath $dir/example_data
				echo "Finished executing $dir, returned status code: $?"
			else
				# pre-preemptively clean up the active directory
				rm -rf $ACTIVE_DIR/*
				# Copy model to the active folder to obscure its name
				cp -r $dir/* $ACTIVE_DIR

				singularity run --contain -B /mnt/scratch -B $ACTIVE_DIR --nv $CONTAINER_EXEC --model_filepath $ACTIVE_DIR/model.pt --result_filepath $ACTIVE_DIR/result.txt --scratch_dirpath $SCRATCH_DIR --examples_dirpath $ACTIVE_DIR/example_data >> $RESULT_DIR/$CONTAINER_NAME.out 2>&1
				echo "Finished executing, returned status code: $?" >> $RESULT_DIR/$CONTAINER_NAME.out 2>&1

				if [[ -f $ACTIVE_DIR/result.txt ]]; then
					# copy result back to real output filename based on model name
					cp $ACTIVE_DIR/result.txt  $RESULT_DIR/$MODEL.txt
				fi
			fi
		fi
	fi
done
