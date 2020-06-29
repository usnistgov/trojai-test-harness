#!/bin/bash

SRC_DIR=/mnt/isgnas/project/ai/trojai/trojai-nist/src/te-scripts/actor_executor

teamName=TestTeam
email=michael.majurski@nist.gov
poc=michael.majurski@nist.gov

# add to the STS
python actor_controller.py --add-actor="$teamName,$email,$poc" --config-file=/mnt/trojainas/round1/config-sts.json --log-file=/mnt/trojainas/round1/sts/actor-manager.log

# add to the ES
python actor_controller.py --add-actor="$teamName,$email,$poc" --config-file=/mnt/trojainas/round1/config-es.json --log-file=/mnt/trojainas/round1/es/actor-manager.log



