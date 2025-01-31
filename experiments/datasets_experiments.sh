#!/bin/bash

DATASETS=(
    "pharmaconer"
    "multicardioner_track2_es"
    "multicardioner_track2_it"
    "multicardioner_track1"
)

MODELS=(
    "phi3.5"
    "granite3.1-dense"
    "phi4"
    "deepseek-r1:14b"
)

PARAGRAPHS_PER_CALL=6

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [ "$model" = "deepseek-r1:14b" ]; then
            calls=0
        else
            calls=$PARAGRAPHS_PER_CALL
        fi
        echo "Running experiment with model=$model and dataset=$dataset"
        python main.py \
            --model "$model" \
            --dataset "$dataset" \
            --sentences-per-call $calls
    done
done