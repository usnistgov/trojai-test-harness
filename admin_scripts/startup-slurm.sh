#!/bin/bash

source vm-list.sh

for vm in $VM_LIST
do
   # Set node as idle in slurm

   scontrol update nodename=gpu-vm-$vm state=idle reason="Startup"
done
