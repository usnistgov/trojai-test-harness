#!/bin/bash
# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

echo "image task" "$@"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
  --result-dir)
    shift
    RESULT_DIR=$1 ;;
  --scratch-dir)
    shift
    SCRATCH_DIR=$1 ;;
  --container-exec)
    shift
    CONTAINER_EXEC=$1 ;;
  --active-dir)
    shift
    ACTIVE_DIR=$1 ;;
  --container-name)
    shift
    CONTAINER_NAME=$1 ;;
  --training-dir)
    shift
    ROUND_TRAINING_DATASET_DIR=$1 ;;
  --source-dir)
    shift
    SOURCE_DATA_DIR=$1 ;;
  *)
    EXTRA_ARGS+=("$1") ;;
  esac
  # Expose next argument
  shift
done

METAPARAMETERS_FILE=/metaparameters.json
METAPARAMETERS_SCHEMA_FILE=/metaparameters_schema.json
LEARNED_PARAMETERS_DIR=/learned_parameters

if [ -z "${SOURCE_DATA_DIR-}" ]; then
  singularity run --contain --bind $ACTIVE_DIR --bind $SCRATCH_DIR \
    --bind $ROUND_TRAINING_DATASET_DIR:$ROUND_TRAINING_DATASET_DIR:ro --nv \
    "$CONTAINER_EXEC" --model_filepath $ACTIVE_DIR/model.pt --result_filepath $ACTIVE_DIR/result.txt \
    --scratch_dirpath $SCRATCH_DIR --examples_dirpath $ACTIVE_DIR/clean-example-data \
    --round_training_dataset_dirpath $ROUND_TRAINING_DATASET_DIR --metaparameters_filepath $METAPARAMETERS_FILE \
    --schema_filepath $METAPARAMETERS_SCHEMA_FILE --learned_parameters_dirpath $LEARNED_PARAMETERS_DIR >> "$RESULT_DIR/$CONTAINER_NAME.out" 2>&1
else
  singularity run --contain --bind $ACTIVE_DIR --bind $SCRATCH_DIR \
    --bind $SOURCE_DATA_DIR:$SOURCE_DATA_DIR:ro --bind $ROUND_TRAINING_DATASET_DIR:$ROUND_TRAINING_DATASET_DIR:ro --nv \
    "$CONTAINER_EXEC" --model_filepath $ACTIVE_DIR/model.pt --result_filepath $ACTIVE_DIR/result.txt \
    --scratch_dirpath $SCRATCH_DIR --examples_dirpath $ACTIVE_DIR/clean-example-data \
    --round_training_dataset_dirpath $ROUND_TRAINING_DATASET_DIR --metaparameters_filepath $METAPARAMETERS_FILE \
    --schema_filepath $METAPARAMETERS_SCHEMA_FILE --learned_parameters_dirpath $LEARNED_PARAMETERS_DIR \
    --source_dataset_dirpath $SOURCE_DATA_DIR >> "$RESULT_DIR/$CONTAINER_NAME.out" 2>&1
fi