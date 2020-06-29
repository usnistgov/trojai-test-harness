#!/bin/bash

source vm-list.sh

for vm in $VM_LIST
do
   # shutdown VM

   virsh shutdown gpu-vm-$vm

done

