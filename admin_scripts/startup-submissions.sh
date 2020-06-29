#!/bin/bash

# Start accepting submissions

sudo -u trojai sed --follow-symlinks -i 's/\"accepting_submissions\": false/\"accepting_submissions\": true/g' /mnt/trojainas/config-es.json
sudo -u trojai sed --follow-symlinks -i 's/\"accepting_submissions\": false/\"accepting_submissions\": true/g' /mnt/trojainas/config-sts.json
