import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_metrics():
    log_files = glob.glob("logs/*.csv")
    if not log_files:
        print("No CSV log files found in logs/")
        return

    plt.figure(figsize=(10, 6))
    
    # Цвета для разных оптимизаторов
    color_map = {
        "adamw": "blue",
        "muon": "red",
        "hybrid": "green",
        "mezo": "purple"
    }

    for file in sorted(log_files):
        # Извлекаем имя оптимизатора из названия файла, например 'adamw_Qwen2.5-0.5B.csv' -> 'adamw'
        opt_name = os.path.basename(file).split('_')[0]
        color = color_map.get(opt_name, "black")
        
        try:
            df = pd.read_csv(file)
            if "step" in df.columns and "loss" in df.columns:
                # Ограничиваем до первых 500 шагов для честного сравнения,
                # если гибрид обучался дольше
                df_subset = df[df["step"] <= 500]
                
                # Считаем экспоненциальное скользящее среднее (EMA) для сглаживания шума
                # alpha=0.1 дает сглаживание примерно за 20 шагов
                smoothed_loss = df_subset["loss"].ewm(alpha=0.1, adjust=False).mean()
                
                plt.plot(df_subset["step"], smoothed_loss, label=f"{opt_name.upper()} (EMA)", color=color, linewidth=2)
                # Полупрозрачный оригинальный loss на фоне
                plt.plot(df_subset["step"], df_subset["loss"], color=color, alpha=0.15)
        except Exception as e:
            print(f"Could not process {file}: {e}")

    plt.title("Training Loss Comparison (First 500 steps, smoothed)")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    output_path = "report/training_loss.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    # Установка стиля (если доступен seaborn)
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        pass
    
    plot_metrics()
