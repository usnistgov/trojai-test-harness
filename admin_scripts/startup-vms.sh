#!/bin/bash

source vm-list.sh

for vm in $VM_LIST
do
   # Start VM

   virsh start gpu-vm-$vm

   if [ $vm == "3b" ]; then ip=2; fi
   if [ $vm == "60" ]; then ip=3; fi
   if [ $vm == "61" ]; then ip=4; fi
   if [ $vm == "86" ]; then ip=5; fi
   if [ $vm == "da" ]; then ip=6; fi
   if [ $vm == "db" ]; then ip=7; fi

   for (( ; ; ))
   do
      sudo -u trojai ssh trojai@192.168.200.$ip nvidia-smi

      if [ $? -eq 0 ]
      then
         break
      fi
   done
done
