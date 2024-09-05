#!/bin/bash

# EXAMPLE SCAN SCRIPT
# Customize per site

SCAN_TEMP=/dev/shm/scan-temp-`uuidgen`

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <Container Name>"
    exit 0
fi

apptainer build --sandbox $SCAN_TEMP $1
clamscan --recursive --infected $SCAN_TEMP
SCAN_RESULT=$?
rm -rf $SCAN_TEMP
exit $SCAN_RESULT
