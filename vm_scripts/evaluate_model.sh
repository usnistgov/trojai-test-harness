#!/bin/bash

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
EXCLUDE_ARGS=()
EXTRA_ARGS=()
ALL_ARGS=$@
while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-dir)
    shift
    MODEL_DIR="$1" ;;
  --gpu-id)
    shift
    GPU_ID="$1" ;;
  --container-path)
    shift
    CONTAINER_EXEC="$1" ;;
  --task-script)
    shift
    TASK_SCRIPT="$1" ;;
  --remote-home)
    shift
    TROJAI_HOME="$1" ;;
  --result-dir)
    shift
    RESULT_DIR="$1" ;;
  --result-prefix)
    shift
    RESULT_PREFIX="$1" ;;
  --remote-scratch)
    shift
    SCRATCH_HOME="$1" ;;
  --rsync-exclude)
    shift
    EXCLUDE_ARGS+=("$1") ;;
  --metaparam-file)
    shift
    METAPARAMETERS_FILE="$1" ;;
  --script-debug)
    SCRIPT_DEBUG=1 ;;
  --submission-filepath)
    shift
    SUBMISSION_FILE="$1" ;;
  *)
    EXTRA_ARGS+=("$1") ;;
  esac
  # Expose next argument
  shift
done

if [ ! -z ${SCRIPT_DEBUG} ]; then
  echo "evaluate model $ALL_ARGS"
fi

# Set positional arguments
set -- "${EXTRA_ARGS[@]}"

echo "Launching task: $TASK_SCRIPT"

ACTIVE_DIR="$SCRATCH_HOME/active_$GPU_ID"

SCRATCH_DIR="$SCRATCH_HOME/container-scratch_$GPU_ID"

MODEL="$(basename "$MODEL_DIR")"

export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU_ID

mkdir -p "$RESULT_DIR"
mkdir -p "$SCRATCH_DIR"

# create the active dir
mkdir -p "$ACTIVE_DIR"

# clean up scratch directory prior to running each model
rm -rf "${SCRATCH_DIR:?}"/*
# pre-preemptively clean up the active directory
rm -rf "${ACTIVE_DIR:?}"/*


# Gather exclude arguments

RSYNC_EXCLUDES=""
for i in "${EXCLUDE_ARGS[@]}"
do
  RSYNC_EXCLUDES+=" --exclude=${i}"
done

# Copy model to the active folder to obscure its name
rsync -ar --prune-empty-dirs --delete $RSYNC_EXCLUDES $MODEL_DIR/* $ACTIVE_DIR

# Create copy of reduced_config.json so we have both reduced-config.json and config.json
if [[ -f "$ACTIVE_DIR/reduced-config.json" ]]; then
  cp "$ACTIVE_DIR/reduced-config.json" $ACTIVE_DIR/"config.json"
fi

# If the container exec does not exist, then we copy from submission_file (useful for local execution only)
if [[ ! -f "$CONTAINER_EXEC" ]]; then
  # Check if submission file is undefined, if it is let us know
  if [ -z "${SUBMISSION_FILE}" ]; then
    echo "Unable to copy to container execution, submission file was not passed into command-line"
  else
    # Copy the submission file into the container executable
    cp "$SUBMISSION_FILE" "$CONTAINER_EXEC"
  fi
fi

# Copy metaparameters file if one is specified
if [ -z "${METAPARAMETERS_FILE-}" ]; then
  METAPARAMETERS_FILE=/metaparameters.json
else
  cp "$METAPARAMETERS_FILE" "$ACTIVE_DIR/metaparameters.json"
  METAPARAMETERS_FILE="$ACTIVE_DIR/metaparameters.json"

fi

echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] [evaluate_model.sh] Starting execution of $MODEL_DIR" >> "$RESULT_DIR/$CONTAINER_NAME.out" 2>&1
if [[ -z "${RESULT_PREFIX-}" ]]; then
  /usr/bin/time -f "execution_time %e" -o "$RESULT_DIR"/"$MODEL"-walltime.txt "$TASK_SCRIPT" --result-dir "$RESULT_DIR" --scratch-dir "$SCRATCH_DIR" --container-exec "$CONTAINER_EXEC" --active-dir "$ACTIVE_DIR" --metaparam-file "$METAPARAMETERS_FILE" "$@"
else
  /usr/bin/time -f "execution_time %e" -o "${RESULT_DIR}/${RESULT_PREFIX}${MODEL}"-walltime.txt "$TASK_SCRIPT" --result-dir "$RESULT_DIR" --scratch-dir "$SCRATCH_DIR" --container-exec "$CONTAINER_EXEC" --active-dir "$ACTIVE_DIR" --metaparam-file "$METAPARAMETERS_FILE" "$@"
fi
echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] [evaluate_model.sh] Finished executing $MODEL_DIR, returned status code: $?"

# copy result back to real output filename based on model name
if [[ -f "$ACTIVE_DIR"/result.txt ]]; then
  # If there is no prefix, then normal model name
  if [[ -z "${RESULT_PREFIX-}" ]]; then
	cp "$ACTIVE_DIR"/result.txt  "${RESULT_DIR}/${MODEL}".txt
  else
    # if there is a prefix add it to the model
    cp "$ACTIVE_DIR"/result.txt  "${RESULT_DIR}/${RESULT_PREFIX}${MODEL}".txt
  fi
fi
