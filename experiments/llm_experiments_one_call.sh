#!/bin/bash

# Run experiments with deepseek-r1 models
models=(
    "llama3.2:1b"
    "llama3.2"
    "phi3.5"
    "granite3.1-dense"
    "falcon3:10b"
    "llama3.2-vision"
    "phi4"
    "mixtral"
    "qwen2.5:32b"
)

for model in "${models[@]}"; do
    echo "Running experiment with $model"
    ollama pull "$model"
    python main.py \
        --model "$model" \
        --dataset "multicardioner_track2_en"
done