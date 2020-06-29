#!/bin/bash

source vm-list.sh

for vm in $VM_LIST
do
   # Attach public (default) network
   virsh attach-interface gpu-vm-$vm network default --live
done
