import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_metrics():
    log_files = glob.glob("logs/*.csv")
    if not log_files:
        print("No CSV log files found in logs/")
        return

    color_map = {
        "adamw": "blue",
        "muon": "red",
        "hybrid": "green",
        "mezo": "purple"
    }

    # === 1. Plot Training Loss ===
    plt.figure(figsize=(10, 6))
    for file in sorted(log_files):
        opt_name = os.path.basename(file).split('_')[0]
        color = color_map.get(opt_name, "black")
        try:
            df = pd.read_csv(file)
            if "step" in df.columns and "loss" in df.columns:
                df_subset = df[df["step"] <= 2000] # Для полноценного прогона
                smoothed_loss = df_subset["loss"].ewm(alpha=0.1, adjust=False).mean()
                plt.plot(df_subset["step"], smoothed_loss, label=f"{opt_name.upper()} (EMA)", color=color, linewidth=2)
                plt.plot(df_subset["step"], df_subset["loss"], color=color, alpha=0.15)
        except Exception as e:
            print(f"Could not process loss for {file}: {e}")

    plt.title("Training Loss Comparison")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/training_loss.png", dpi=300)
    plt.close()

    # === 2. Plot VRAM Usage ===
    plt.figure(figsize=(10, 6))
    vram_stats = []
    for file in sorted(log_files):
        opt_name = os.path.basename(file).split('_')[0]
        color = color_map.get(opt_name, "black")
        try:
            df = pd.read_csv(file)
            if "step" in df.columns and "vram_mb" in df.columns:
                # Фильтруем нули (если запуск был на CPU или без учета VRAM)
                df_valid = df[df["vram_mb"] > 0]
                if not df_valid.empty:
                    df_subset = df_valid[df_valid["step"] <= 2000]
                    plt.plot(df_subset["step"], df_subset["vram_mb"], label=opt_name.upper(), color=color, linewidth=2)
                    vram_stats.append({"Optimizer": opt_name.upper(), "Peak VRAM (MB)": int(df_valid["vram_mb"].max())})
        except Exception as e:
            print(f"Could not process vram for {file}: {e}")

    if vram_stats:
        plt.title("Peak VRAM Allocation Comparison")
        plt.xlabel("Training Steps")
        plt.ylabel("Allocated VRAM (MB)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig("report/vram_usage.png", dpi=300)
        
        print("\n--- Peak VRAM Usage Table ---")
        stats_df = pd.DataFrame(vram_stats)
        print(stats_df.to_string(index=False))
        print("-----------------------------\n")
    plt.close()

if __name__ == "__main__":
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        pass
    plot_metrics()
