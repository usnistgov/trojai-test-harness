#!/bin/bash

source /home/trojai/miniconda3/bin/activate trojai_evaluate

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
  --models-dirpath)
    shift
    MODELS_DIRPATH=$1 ;;
  --submission-type)
    shift
    SUBMISSION_TYPE=$1 ;;
  --submission-filepath)
    shift
    SUBMISSION_FILEPATH=$1 ;;
  --home-dirpath)
    shift
    HOME_DIRPATH=$1 ;;
  --result-dirpath)
    shift
    RESULT_DIRPATH=$1 ;;
  --scratch-dirpath)
    shift
    SCRATCH_DIRPATH=$1 ;;
  --training-dataset-dirpath)
    shift
    TRAINING_DATASET_DIRPATH=$1 ;;
  --metaparameters-filepath)
    shift
    METAPARAMETERS_FILEPATH=$1 ;;
  --rsync-excludes)
    shift
    TROJAI_TEST_HARNESS_DIRPATH=$1 ;;
  --python-exec)
    shift
    PYTHON_EXEC=$1 ;;
  --task-executor-filepath)
    shift
    TASK_EXECUTOR_FILEPATH=$1 ;;
  --is-local)
    EXECUTE_LOCAL=1 ;;
  --custom-home)
    shift
    CUSTOM_HOME=$1 ;;
  --custom-scratch)
    shift
    CUSTOM_SCRATCH=$1 ;;
  *)
    EXTRA_ARGS+=("$1") ;;
  esac
  # Expose next argument
  shift
done

echo "Extra args: $EXTRA_ARGS"