#!/bin/bash

# Создаем папки
mkdir -p logs checkpoints results report

echo "=== 1. Full Training AdamW ==="
# Эффективный batch size = 4, 500000 шагов (всего 2000000 проходов модели)
uv run python -m src.training.trainer --optimizer adamw --lr 5e-5 --grad_accum_steps 4 --max_steps 500000 --epochs 5

echo "=== 2. Full Training Muon ==="
# Для Muon ставим базовый lr=1e-4, а для бэкап-весов (эмбеддингов) adamw_lr=5e-5
uv run python -m src.training.trainer --optimizer muon --lr 1e-4 --adamw_lr 5e-5 --grad_accum_steps 4 --max_steps 500000 --epochs 5

echo "=== 3. Full Training Hybrid ==="
# Аналогично для гибрида (Muon на первые 12 слоев, AdamW на остальные)
uv run python -m src.training.trainer --optimizer hybrid --lr 1e-4 --adamw_lr 5e-5 --grad_accum_steps 4 --max_steps 500000 --epochs 5

echo "=== 4. Full Training MeZO ==="
# MeZO без градиентов, ставим шаг чуть больше, чтобы попытаться сдвинуть loss
uv run python -m src.training.trainer --optimizer mezo --lr 1e-5 --grad_accum_steps 4 --max_steps 500000 --epochs 5

echo "=== 5. Generating Metrics Plots (Loss & VRAM) ==="
uv run python scripts/plot_metrics.py

echo "Full training and memory profiling finished!"
