#!/bin/bash

SRC_DIR=/mnt/isgnas/project/ai/trojai/trojai-nist/src/te-scripts/actor_executor

email=michael.majurski@nist.gov

# remove from the STS
python actor_controller.py --remove-actor="$email" --config-file=/mnt/trojainas/round1/config-sts.json --log-file=/mnt/trojainas/round1/sts/actor-manager.log

# remove the ES
python actor_controller.py --remove-actor="$email" --config-file=/mnt/trojainas/round1/config-es.json --log-file=/mnt/trojainas/round1/es/actor-manager.log

