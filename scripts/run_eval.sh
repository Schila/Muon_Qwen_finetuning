#!/bin/bash

# Создаем папки
mkdir -p logs checkpoints results

echo "=== 1. Training AdamW ==="
python -m src.training.trainer --optimizer adamw --lr 1e-5 --max_steps 15000

echo "=== 2. Training Muon ==="
python -m src.training.trainer --optimizer muon --lr 1e-4 --max_steps 15000

# Если вы хотите обучить гибридную модель (опционально, раскомментируйте)
# echo "=== 3. Training Hybrid ==="
# python -m src.training.trainer --optimizer hybrid --lr 0.01 --max_steps 500

echo "=== 4. Evaluation for AdamW ==="
# --batch_size auto автоматически подберет максимальный размер батча для вашей видеокарты
lm_eval --model hf \
    --model_args pretrained=checkpoints/adamw_Qwen2.5-0.5B \
    --tasks piqa,arc_easy,arc_challenge,winogrande,hellaswag \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/adamw_eval.json

echo "=== 5. Evaluation for Muon ==="
lm_eval --model hf \
    --model_args pretrained=checkpoints/muon_Qwen2.5-0.5B \
    --tasks piqa,arc_easy,arc_challenge,winogrande,hellaswag \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/muon_eval.json

echo "=== 6. Baseline Evaluation (Untrained Qwen2.5-0.5B) ==="
# Это позволит нам сравнить обученные модели с оригинальной моделью, чтобы понять, стал ли loss лучше на 500 шагах.
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B \
    --tasks piqa,arc_easy,arc_challenge,winogrande,hellaswag \
    --device cuda:0 \
    --batch_size auto \
    --output_path results/baseline_eval.json

echo "All evaluations finished! Results saved in results/ folder."
