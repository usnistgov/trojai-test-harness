#!/bin/bash

PYTHONPATH=/home/trojai/trojai-test-harness /home/trojai/test-env/bin/python3 -u /home/trojai/trojai-test-harness/actor_executor/check-and-launch-actors.py --config-file /mnt/trojainas/config-es.json
PYTHONPATH=/home/trojai/trojai-test-harness /home/trojai/test-env/bin/python3 -u /home/trojai/trojai-test-harness/actor_executor/check-and-launch-actors.py --config-file /mnt/trojainas/config-sts.json
PYTHONPATH=/home/trojai/trojai-test-harness /home/trojai/test-env/bin/python3 -u /home/trojai/trojai-test-harness/actor_executor/execute_html_update.py --no-push-html --config-file /mnt/trojainas/config-es.json
PYTHONPATH=/home/trojai/trojai-test-harness /home/trojai/test-env/bin/python3 -u /home/trojai/trojai-test-harness/actor_executor/execute_html_update.py --config-file /mnt/trojainas/config-sts.json