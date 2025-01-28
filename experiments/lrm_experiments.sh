#!/bin/bash

# Run experiments with deepseek-r1 models
models=(
    "deepseek-r1:7b"
    "deepseek-r1:8b"
    "deepseek-r1:14b"
    "deepseek-r1:32b"
)

for model in "${models[@]}"; do
    echo "Running experiment with $model"
    python main.py \
        --model "$model" \
        --dataset "multicardioner_track2_en"
done