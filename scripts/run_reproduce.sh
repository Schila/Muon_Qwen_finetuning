#!/bin/bash

# Reproduce all experiments for the comparative study

mkdir -p logs checkpoints results

echo "=== 1. Training AdamW Baseline ==="
uv run python -m src.training.trainer --optimizer adamw --lr 1e-5 --max_steps 500

echo "=== 2. Training Muon (Full) ==="
# Using a reduced LR for fine-tuning stability
uv run python -m src.training.trainer --optimizer muon --lr 1e-4 --adamw_lr 1e-5 --max_steps 500

echo "=== 3. Training Hybrid (Muon + AdamW) ==="
# Muon on first half, AdamW on second half
uv run python -m src.training.trainer --optimizer hybrid --lr 1e-4 --adamw_lr 1e-5 --max_steps 500

echo "=== 4. Training MeZO (Zeroth-Order Challenge) ==="
# MeZO requires more steps to see any progress
uv run python -m src.training.trainer --optimizer mezo --lr 1e-6 --max_steps 15000

echo "=== 5. Running Evaluation Harness ==="
# Evaluate all models
for model in adamw muon hybrid mezo; do
    echo "Evaluating $model..."
    uv run lm_eval --model hf \
        --model_args pretrained=checkpoints/${model}_Qwen2.5-0.5B \
        --tasks piqa,arc_easy,arc_challenge,winogrande,hellaswag \
        --device cuda:0 \
        --batch_size auto \
        --output_path results/${model}_eval.json
done

echo "=== 6. Generating Plots ==="
uv run python scripts/plot_metrics.py

echo "Reproductions complete. Check report/ and results/ folders."
