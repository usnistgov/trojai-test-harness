#!/bin/bash

IPs="192.168.200.4 192.168.200.7"
MODEL_DIR="/mnt/trojainas/datasets/round1/round1-dataset-sts/models/"

for ip in $IPs; do
	echo "Deleting old models on $ip"
	ssh trojai@$ip "rm -rf /home/trojai/data/*"
	echo "Copying to $ip"
	rsync -ar --prune-empty-dirs $MODEL_DIR trojai@$ip:/home/trojai/data/
	rsync -ar /mnt/isgnas/project/ai/trojai/trojai-test-harness/vm_scripts/evaluate_models.sh trojai@$ip:/home/trojai/evaluate_models.sh
	echo "Done copying to $ip"
done


IPs="192.168.200.2 192.168.200.3 192.168.200.5 192.168.200.6"
MODEL_DIR="/mnt/trojainas/datasets/round1/round1-dataset-test/models/"

for ip in $IPs; do
	echo "Deleting old models on $ip"
	ssh trojai@$ip "rm -rf /home/trojai/data/*"
	echo "Copying to $ip"
	rsync -ar --prune-empty-dirs $MODEL_DIR trojai@$ip:/home/trojai/data/
	rsync -ar /mnt/isgnas/project/ai/trojai/trojai-test-harness/vm_scripts/evaluate_models.sh trojai@$ip:/home/trojai/evaluate_models.sh
	echo "Done copying to $ip"
done

