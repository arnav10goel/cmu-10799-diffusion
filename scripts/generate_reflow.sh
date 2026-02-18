#!/bin/bash
# Usage: ./scripts/generate_reflow.sh <CHECKPOINT> <CONFIG> <NAME>

CHECKPOINT=$1
CONFIG=$2
NAME=${3:-"teacher"}

if [ -z "$CHECKPOINT" ] || [ -z "$CONFIG" ]; then
    echo "Usage: $0 <checkpoint_path> <config_path> [output_name]"
    echo "Example: $0 checkpoints/fm_final.pt configs/flow_matching.yaml v1"
    exit 1
fi

# Output directory
OUTPUT_DIR="data/distillation_${NAME}"

echo "=========================================="
echo "Starting Reflow Data Generation"
echo "Teacher: $CHECKPOINT"
echo "Config:  $CONFIG"
echo "Output:  $OUTPUT_DIR"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

CUDA_VISIBLE_DEVICES=7 python src/methods/generate_reflow_teacher_data.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 100000 \
    --batch_size 128 \
    --steps 100 \
    --device cuda

echo "Generation Complete."