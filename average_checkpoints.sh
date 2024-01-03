#!/bin/bash

VIRTUALENV=/sc04a2/users/jiblaing/fairseq-venv/bin
inputs=(checkpoints/checkpoint.es-en.pt checkpoints/checkpoint.en-nl.pt checkpoints/checkpoint.en-es.pt checkpoints/checkpoint.nl-en.pt)
output=checkpoints/checkpoint_avg.pt

# Activate virtual environment
source ${VIRTUALENV}/activate

python average_checkpoints.py \
    --inputs ${inputs[1]} ${inputs[2]} ${inputs[3]} ${inputs[4]} \
    --output ${output}