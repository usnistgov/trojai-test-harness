#!/bin/bash
teamName=$1
submissionPath=$2
resultDir=$3
configFile=$4
email=$5
holdoutConfigFile=$6
pythonExecutorScript=$7

#echo $SLURM_JOB_NODELIST_PACK_GROUP_0  # host
#echo $SLURM_JOB_NODELIST_PACK_GROUP_1  # vm

PYTHONEXEC=/home/trojai/test-env/bin/python3

$PYTHONEXEC -u $pythonExecutorScript --team-name $teamName --submission-filepath $submissionPath --result-dir $resultDir --config-file $configFile --team-email $email --vm-name $SLURM_JOB_NODELIST_PACK_GROUP_1 --holdout-config-file $holdoutConfigFile
