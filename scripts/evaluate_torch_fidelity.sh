#!/bin/bash
# =============================================================================
# Torch-Fidelity Evaluation Script
# =============================================================================
#
# Usage:
#   ./scripts/evaluate_torch_fidelity.sh \
#       --checkpoint checkpoints/ddpm/ddpm_final.pt \
#       --method ddpm \
#       --sampler ddim \
#       --num-steps 50 \
#       --dataset-path data/celeba \
#       --metrics kid
#
# =============================================================================

set -e

# Defaults
METHOD="flow_matching"
SAMPLER="default"
CHECKPOINT="/home/arnavgoe/10799-Diffusion/cmu-10799-diffusion/logs/reflow_unet_distillation/checkpoints/flow_matching_final.pt"
DATASET_PATH="/home/arnavgoe/10799-Diffusion/cmu-10799-diffusion/data/train/images"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=128
NUM_STEPS=""
GENERATED_DIR=""
CACHE_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --sampler) SAMPLER="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

if [ -z "$NUM_STEPS" ]; then
    NUM_STEPS=$(python - "$CHECKPOINT" "$METHOD" <<'PY'
import sys
import torch

checkpoint_path = sys.argv[1]
method = sys.argv[2]

ckpt = torch.load(checkpoint_path, map_location='cpu')
cfg = ckpt.get('config', {})

if method == 'ddpm':
    print(cfg.get('ddpm', {}).get('num_timesteps', 1000))
else:
    print(cfg.get('sampling', {}).get('num_steps', 100))
PY
)
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "Warning: Dataset path $DATASET_PATH does not exist."
fi

CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
# Include sampler in folder name so DDIM results don't overwrite DDPM results
SAMPLER_SUFFIX=""
if [ "$SAMPLER" != "default" ]; then
    SAMPLER_SUFFIX="_${SAMPLER}"
fi
STEP_SUFFIX=${NUM_STEPS:-"default"}

GENERATED_DIR="${CHECKPOINT_DIR}/eval_samples_${METHOD}${SAMPLER_SUFFIX}_${STEP_SUFFIX}"
CACHE_DIR="${CHECKPOINT_DIR}/fidelity_cache"

echo "=========================================="
echo "Torch-Fidelity Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Method: $METHOD"
echo "Sampler: $SAMPLER"
echo "Steps: ${NUM_STEPS:-Default}"
echo "Dataset: $DATASET_PATH"
echo "Metrics: $METRICS"
echo "Num samples: $NUM_SAMPLES"
echo "Output: $GENERATED_DIR"
echo "=========================================="

# Step 1: Generate samples
echo ""
echo "[1/2] Generating samples..."
rm -rf "$GENERATED_DIR"

SAMPLE_CMD="python sample.py \
    --checkpoint $CHECKPOINT \
    --method $METHOD \
    --sampler $SAMPLER \
    --output_dir $GENERATED_DIR \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE"

if [ -n "$NUM_STEPS" ]; then
    SAMPLE_CMD="$SAMPLE_CMD --num_steps $NUM_STEPS"
fi

echo "Running: $SAMPLE_CMD"
eval $SAMPLE_CMD

# Step 2: Run fidelity
echo ""
echo "[2/2] Computing metrics..."
mkdir -p "$CACHE_DIR"

FIDELITY_CMD="fidelity --gpu 0 --batch-size $BATCH_SIZE --cache-root $CACHE_DIR \
    --input1 $GENERATED_DIR --input2 $DATASET_PATH"

[[ "$METRICS" == *"fid"* ]] && FIDELITY_CMD="$FIDELITY_CMD --fid"
[[ "$METRICS" == *"kid"* ]] && FIDELITY_CMD="$FIDELITY_CMD --kid"
[[ "$METRICS" == *"is"* ]] && FIDELITY_CMD="$FIDELITY_CMD --isc"

echo "Running: $FIDELITY_CMD"
eval $FIDELITY_CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="