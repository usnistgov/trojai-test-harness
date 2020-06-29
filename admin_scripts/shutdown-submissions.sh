#!/bin/bash

# Stop accepting submissions

sudo -u trojai sed --follow-symlinks -i 's/\"accepting_submissions\": true/\"accepting_submissions\": false/g' /mnt/trojainas/config-es.json
sudo -u trojai sed --follow-symlinks -i 's/\"accepting_submissions\": true/\"accepting_submissions\": false/g' /mnt/trojainas/config-sts.json
