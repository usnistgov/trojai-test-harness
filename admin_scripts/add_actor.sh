#!/bin/bash

conda activate scripts
cd /mnt/isgnas/project/ai/trojai/trojai-test-harness/actor_executor

teamName=TestTeam
email=
poc=

# add to the STS
python actor_controller.py --add-actor="$teamName,$email,$poc" --config-file=/mnt/trojainas/round2/config-sts.json --log-file=/mnt/trojainas/round2/sts/actor-manager.log

# add to the ES
python actor_controller.py --add-actor="$teamName,$email,$poc" --config-file=/mnt/trojainas/round2/config-es.json --log-file=/mnt/trojainas/round2/es/actor-manager.log



