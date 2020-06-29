#!/bin/bash
teamName=$1
submissionDir=$2
resultDir=$3
configFile=$4
email=$5

#echo $SLURM_JOB_NODELIST_PACK_GROUP_0  # host
#echo $SLURM_JOB_NODELIST_PACK_GROUP_1  # vm

PYTHONEXEC=/home/trojai/test-env/bin/python3

$PYTHONEXEC -u /mnt/isgnas/project/ai/trojai/trojai-nist/src/te-scripts/actor_executor/vm-executor.py --team-name $teamName --submission-dir $submissionDir --result-dir $resultDir --config-file $configFile --team-email $email --vm-name $SLURM_JOB_NODELIST_PACK_GROUP_1

