#!/bin/bash

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


CONTAINER_NAME=$1
QUEUE_NAME=$2
dir=$3

ACTIVE_DIR="/home/trojai/active_$SLURM_JOB_ID"

CONTAINER_EXEC="/mnt/scratch/$CONTAINER_NAME"
RESULT_DIR=/mnt/scratch/results
SCRATCH_DIR="/mnt/scratch/container-scratch_$SLURM_JOB_ID"
SOURCE_DATA_DIR=/home/trojai/source_data

TOKENIZER_DIR=/home/trojai/tokenizers

ROUND_TRAINING_DATASET_DIR=/home/trojai/round_training_dataset

METAPARAMETERS_FILE=/metaparameters.json
METAPARAMETERS_SCHEMA_FILE=/metaparameters_schema.json
LEARNED_PARAMETERS_DIR=/learned_parameters

MODEL="$(basename $dir)"

# create needed directories if they do not exist
mkdir -p $RESULT_DIR
mkdir -p $SCRATCH_DIR
mkdir -p $ACTIVE_DIR

# theoretically these should be empty, but clean them anyway in case there is something leftover from an old run
rm -rf $SCRATCH_DIR/*
rm -rf $ACTIVE_DIR/*

# Copy model to the active folder to obscure its name
cp -r $dir/* $ACTIVE_DIR

# Determine which embedding, tokenizer, and cls to use
TOKENIZER_FILENAME=`cat $dir/config.json | python3 -c "import sys, json; print(json.load(sys.stdin)['tokenizer_filename'])"`
TOKENIZER_FILEPATH=$TOKENIZER_DIR/$TOKENIZER_FILENAME

if [[ "$QUEUE_NAME" == "sts" ]]; then
  echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] [evaluate_model.sh] Starting execution of $dir"

	singularity run --contain --bind $ACTIVE_DIR --bind $RESULT_DIR --bind $SCRATCH_DIR --bind $TOKENIZER_DIR:$TOKENIZER_DIR:ro  --bind $SOURCE_DATA_DIR:$SOURCE_DATA_DIR:ro --bind $ROUND_TRAINING_DATASET_DIR:$ROUND_TRAINING_DATASET_DIR:ro --nv "$CONTAINER_EXEC" --model_filepath $ACTIVE_DIR/model.pt --result_filepath $RESULT_DIR/$MODEL.txt --scratch_dirpath $SCRATCH_DIR --examples_dirpath $ACTIVE_DIR/example_data --tokenizer_filepath $TOKENIZER_FILEPATH --round_training_dataset_dirpath $ROUND_TRAINING_DATASET_DIR --metaparameters_filepath $METAPARAMETERS_FILE --schema_filepath $METAPARAMETERS_SCHEMA_FILE --learned_parameters_dirpath $LEARNED_PARAMETERS_DIR
	echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] [evaluate_model.sh] Finished executing $dir, returned status code: $?"
	#echo "Finished executing $dir, returned status code: $?"
else
  echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] [evaluate_model.sh] Starting execution of $dir" >> "$RESULT_DIR/$CONTAINER_NAME.out" 2>&1

	/usr/bin/time -f "execution_time %e" -o $RESULT_DIR/$MODEL-walltime.txt singularity run --contain --bind $ACTIVE_DIR --bind $SCRATCH_DIR --bind $TOKENIZER_DIR:$TOKENIZER_DIR:ro --bind $SOURCE_DATA_DIR:$SOURCE_DATA_DIR:ro --bind $ROUND_TRAINING_DATASET_DIR:$ROUND_TRAINING_DATASET_DIR:ro --nv "$CONTAINER_EXEC" --model_filepath $ACTIVE_DIR/model.pt --result_filepath $ACTIVE_DIR/result.txt --scratch_dirpath $SCRATCH_DIR --examples_dirpath $ACTIVE_DIR/example_data --tokenizer_filepath $TOKENIZER_FILEPATH --round_training_dataset_dirpath $ROUND_TRAINING_DATASET_DIR --metaparameters_filepath $METAPARAMETERS_FILE --schema_filepath $METAPARAMETERS_SCHEMA_FILE --learned_parameters_dirpath $LEARNED_PARAMETERS_DIR >> "$RESULT_DIR/$CONTAINER_NAME.out" 2>&1
	echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] [evaluate_model.sh] Finished executing, returned status code: $?" >> "$RESULT_DIR/$CONTAINER_NAME.out" 2>&1
	#echo "Finished executing, returned status code: $?" >> "$RESULT_DIR/$CONTAINER_NAME.out" 2>&1
fi

# copy result back to real output filename based on model name
if [[ -f $ACTIVE_DIR/result.txt ]]; then
	cp $ACTIVE_DIR/result.txt  $RESULT_DIR/$MODEL.txt
fi

# remove directories used for this run
rm -rf $SCRATCH_DIR
rm -rf $ACTIVE_DIR