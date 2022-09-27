#!/bin/bash
# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
  --team-name)
    shift
    TEAM_NAME=$1 ;;
  --team-email)
    shift
    TEAM_EMAIL=$1 ;;
  --submission-filepath)
    shift
    SUBMISSION_FILEPATH=$1 ;;
  --result-dirpath)
    shift
    RESULT_DIRPATH=$1 ;;
  --trojai-config-filepath)
    shift
    TROJAI_CONFIG_FILEPATH=$1 ;;
  --leaderboard-name)
    shift
    LEADERBOARD_NAME=$1 ;;
  --data-split-name)
    shift
    DATA_SPLIT_NAME=$1 ;;
  --trojai-test-harness-dirpath)
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


echo $SLURM_JOB_NODELIST_PACK_GROUP_0  # host
echo $SLURM_JOB_NODELIST_PACK_GROUP_1  # vm

if [ -z "${EXECUTE_LOCAL-}" ]; then
  echo "Normal execution"
  PYTHONPATH=$TROJAI_TEST_HARNESS_DIRPATH $PYTHON_EXEC -u $TASK_EXECUTOR_FILEPATH --team-name $TEAM_NAME --team-email $TEAM_EMAIL --container-filepath $SUBMISSION_FILEPATH --result-dirpath $RESULT_DIRPATH --trojai-config-filepath $TROJAI_CONFIG_FILEPATH --leaderboard-name $LEADERBOARD_NAME --data-split-name $DATA_SPLIT_NAME --vm-name $SLURM_JOB_NODELIST_PACK_GROUP_1
else
  echo "Executing locally"
  if [ -z "${CUSTOM_HOME-}" ]; then
    CUSTOM_HOME=$HOME
  fi

  if [ -z "${CUSTOM_SCRATCH-}" ]; then
    CUSTOM_SCRATCH="/scratch/$USER"
  fi
  PYTHONPATH=$TROJAI_TEST_HARNESS_DIRPATH $PYTHON_EXEC -u $TASK_EXECUTOR_FILEPATH --team-name $TEAM_NAME --team-email $TEAM_EMAIL --container-filepath $SUBMISSION_FILEPATH --result-dirpath $RESULT_DIRPATH --trojai-config-filepath $TROJAI_CONFIG_FILEPATH --leaderboard-name $LEADERBOARD_NAME --data-split-name $DATA_SPLIT_NAME --vm-name "local" --custom-remote-home $CUSTOM_HOME --custom-remote-scratch $CUSTOM_SCRATCH
fi

