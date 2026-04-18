#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep over learning rate and gradient accumulation steps.
# Model checkpoints are disabled; only training statistics are saved.

LEARNING_RATES=(1e-5 5e-5 1e-4)
GRAD_ACCUM_STEPS=(10 20 40)

OUTPUT_DIR="logs/sweep"
mkdir -p "$OUTPUT_DIR"

for LR in "${LEARNING_RATES[@]}"; do
    for GRAD_ACCUM in "${GRAD_ACCUM_STEPS[@]}"; do
        RUN_NAME="lr_${LR}_grad_accum_${GRAD_ACCUM}"
        OUTPUT_PATH="${OUTPUT_DIR}/${RUN_NAME}.json"

        echo "========================================="
        echo "Run: ${RUN_NAME}"
        echo "  lr=${LR}  grad_accum=${GRAD_ACCUM}"
        echo "========================================="

        python train_PRM.py \
            --lr "$LR" \
            --gradient_accumulation_steps "$GRAD_ACCUM" \
            --run_name "$RUN_NAME" \
            --output "$OUTPUT_PATH" \
            --no_checkpoint \
            --epochs 1

        echo "Saved statistics → ${OUTPUT_PATH}"
        echo
    done
done

echo "Sweep complete. Results in ${OUTPUT_DIR}/"
