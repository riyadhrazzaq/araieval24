#!/bin/bash

set -e

WORK_DIR="checkpoints/classweight-adamw"
INPUT_FILE="task1/data/task1_train.jsonl"
CHECKPOINT="$WORK_DIR/model_best.pt"
OUTPUT_FILE="$WORK_DIR/task1_train.jsonl.hyp"

# generate
python task1/src/generate.py \
$INPUT_FILE \
$OUTPUT_FILE \
$CHECKPOINT

# score
python -m task1.scorer.task1 \
-g $INPUT_FILE \
-p $OUTPUT_FILE