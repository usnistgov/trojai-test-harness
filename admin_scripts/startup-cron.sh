#!/bin/bash

# Start cron job

sudo sed -i 's/^#\*/\*/g' /etc/cron.d/actor-executor
sudo sed -i 's/^#\*/\*/g' /etc/cron.d/actor-executor-test
