#!/bin/bash

CONTAINER_NAME=$1
QUEUE_NAME=$2
MODEL_DIR=$3

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
