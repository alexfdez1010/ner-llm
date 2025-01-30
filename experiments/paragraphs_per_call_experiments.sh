#!/bin/bash

for i in {0..10}; do
    echo "Running experiment with sentences per call=$i"
    python main.py \
        --model "phi4" \
        --dataset "multicardioner_track2_en" \
        --sentences-per-call $i
done