import argparse
import time
import torch
import pandas as pd
from loguru import logger
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from src.data.dataset import get_dataloaders
from src.models.optimizers import get_optimizer
from src.models.mezo import MeZO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "muon", "hybrid", "mezo"], default="adamw")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adamw_lr", type=float, default=None, help="Specific LR for AdamW branch (useful for Muon/Hybrid)")
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    # Логирование
    log_file = f"logs/{args.optimizer}_{args.model_name.split('/')[-1]}.csv"
    logger.add(f"logs/train_{args.optimizer}.log")
    history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Загрузка модели и данных
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    train_dataloader, tokenizer = get_dataloaders(args.model_name, args.batch_size)
    
    # Инициализация оптимизатора
    if args.optimizer == "mezo":
        optimizer = MeZO(model, lr=args.lr)
    else:
        optimizer = get_optimizer(model, args)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=args.max_steps // args.grad_accum_steps)

    model.train()
    step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        for batch in train_dataloader:
            if step >= args.max_steps:
                break
            
            batch = batch.to(device)
            step_start = time.time()
            
            if args.optimizer == "mezo":
                # MeZO не использует backward()
                loss_val = optimizer.step(batch)
            else:
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()
                
                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == args.max_steps:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                loss_val = outputs.loss.item()

            step_time = time.time() - step_start
            current_lr = args.lr if args.optimizer == "mezo" else scheduler.get_last_lr()[0]
            
            # Получение статистики памяти
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
            
            # Сохранение логов
            log_entry = {
                "step": step,
                "loss": loss_val,
                "lr": current_lr,
                "step_time": step_time,
                "total_time": time.time() - start_time,
                "vram_mb": vram_mb
            }
            history.append(log_entry)
            
            if step % 10 == 0:
                logger.info(f"Step {step} | Loss: {loss_val:.4f} | LR: {current_lr:.2e} | VRAM: {vram_mb:.0f} MB | Time: {step_time:.2f}s")
            
            step += 1
            
        if step >= args.max_steps:
            break

    # Сохранение истории в CSV
    pd.DataFrame(history).to_csv(log_file, index=False)
    logger.info(f"Training finished. Logs saved to {log_file}")

    # Сохранение модели и токенизатора
    output_dir = f"checkpoints/{args.optimizer}_{args.model_name.split('/')[-1]}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
