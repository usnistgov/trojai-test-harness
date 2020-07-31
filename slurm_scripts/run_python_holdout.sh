#!/bin/bash
teamName=$1
submissionPath=$2
resultDir=$3
configFile=$4
holdoutConfigFile=$5
pythonExecutorScript=$6

#echo $SLURM_JOB_NODELIST_PACK_GROUP_0  # host
#echo $SLURM_JOB_NODELIST_PACK_GROUP_1  # vm

PYTHONEXEC=/home/trojai/test-env/bin/python3

$PYTHONEXEC -u $pythonExecutorScript --team-name $teamName --submission-filepath $submissionPath --result-dir $resultDir --config-file $configFile --vm-name $SLURM_JOB_NODELIST_PACK_GROUP_1 --holdout-config-file $holdoutConfigFile

