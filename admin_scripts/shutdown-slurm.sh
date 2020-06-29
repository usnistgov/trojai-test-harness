#!/bin/bash

source vm-list.sh

for vm in $VM_LIST
do
   # Set node as draining in slurm

   scontrol update nodename=gpu-vm-$vm state=drain reason="Maintenance"
done
