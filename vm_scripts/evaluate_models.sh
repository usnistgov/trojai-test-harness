#!/bin/bash

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

EXTRA_ARGS=()
SUBSET_MODELS=()
#echo "evaluate models" "$@"
while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-dir)
    shift
    MODEL_DIR="$1" ;;
  --evaluate-model-filepath)
    shift
    EVALUATE_SCRIPT="$1" ;;
  --subset-model-id)
    shift
    SUBSET_MODELS+=("$1") ;;
  *)
    EXTRA_ARGS+=("$1") ;;
  esac
  # Expose next argument
  shift
done

# Set positional arguments
set -- "${EXTRA_ARGS[@]}"

if ! command -v nvidia-smi &> /dev/null
then
  echo "Failed to find nvidia-smi command to get GPUs"
  exit 1
fi

NUM_GPUS=`nvidia-smi --list-gpus | wc -l`



NVSMI_STATUS=$?

# initialize process ids
for ((GPU_ID=0;GPU_ID<NUM_GPUS;GPU_ID++))
do
	PROCESS_IDS[$GPU_ID]=0
done

re='^[0-9]+$'

if ! [[ $NUM_GPUS =~ $re ]] ; then

  exit 1
fi

# find all the 'id-' model files and shuffle their iteration order
for dir in `find "$MODEL_DIR" -maxdepth 1 -type d | shuf`
do
	# check that the directory is not the root MODEL_DIR
	if [ "$dir" != "$MODEL_DIR" ]; then
		# check that the directory starts with "id"
		MODEL="$(basename $dir)"

		if [[ "$MODEL" == id* ]] ; then
		    # Check if we have subset the model IDs
		    if [ ${#SUBSET_MODELS[@]} -gt 0 ] ; then
		      # check if MODEL does not exists in subset models, if it is not in there, then we skip it
		      if [[ ! "${SUBSET_MODELS[*]}" =~ ${MODEL} ]] ; then
		        continue
              fi
		    fi


			# find a free GPU
			FREE_GPU_ID=-1
			until [ $FREE_GPU_ID != -1 ]
			do
				for ((GPU_ID=0;GPU_ID<NUM_GPUS;GPU_ID++))
				do
					# check if GPU's process is no longer running
					if [ ! -d /proc/${PROCESS_IDS[$GPU_ID]} ]; then
						FREE_GPU_ID=$GPU_ID
					fi
				done

				sleep 1s
			done

			# launch the job with the remaining argument
			$EVALUATE_SCRIPT --model-dir "$dir" --gpu-id "$FREE_GPU_ID" "$@" &
			PROCESS_IDS[$FREE_GPU_ID]=$!
		fi
	fi
done
