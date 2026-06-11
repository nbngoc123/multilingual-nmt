import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    history_path = "outputs/training_history.json"
    if not os.path.exists(history_path):
        print(f"❌ Không tìm thấy file {history_path}.")
        print("Bạn cần thêm đoạn code lưu log_history sau khi chạy trainer.train()!")
        return

    # 1. Đọc dữ liệu lịch sử
    with open(history_path, "r", encoding="utf-8") as f:
        log_history = json.load(f)

    # 2. Phân loại dữ liệu
    train_logs = []
    eval_logs = []
    
    for log in log_history:
        if "loss" in log and "step" in log:
            # Training log
            train_logs.append({
                "step": log["step"],
                "loss": log["loss"],
                "learning_rate": log.get("learning_rate", 0)
            })
        elif "eval_loss" in log and "step" in log:
            # Evaluation log
            eval_logs.append({
                "step": log["step"],
                "epoch": log.get("epoch", 0),
                "eval_loss": log["eval_loss"],
                "eval_bleu": log.get("eval_bleu", 0)
            })

    df_train = pd.DataFrame(train_logs)
    df_eval = pd.DataFrame(eval_logs)

    if df_train.empty and df_eval.empty:
        print("❌ File lịch sử rỗng. Mô hình chưa train được bước nào.")
        return

    # 3. Vẽ biểu đồ
    os.makedirs("outputs", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Biểu đồ quá trình Huấn luyện (Training History)", fontsize=16, fontweight='bold', y=1.05)

    # Đồ thị 1: Training Loss
    if not df_train.empty:
        sns.lineplot(ax=axes[0], data=df_train, x="step", y="loss", color="blue", linewidth=2)
        axes[0].set_title("Training Loss theo Step", fontsize=14)
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Loss")

    # Đồ thị 2: Evaluation Loss
    if not df_eval.empty:
        # Dùng epoch làm trục x nếu có, nếu không dùng step
        x_col = "epoch" if "epoch" in df_eval.columns and df_eval["epoch"].nunique() > 1 else "step"
        sns.lineplot(ax=axes[1], data=df_eval, x=x_col, y="eval_loss", color="red", marker="o", linewidth=2)
        axes[1].set_title(f"Validation Loss theo {x_col.capitalize()}", fontsize=14)
        axes[1].set_xlabel(x_col.capitalize())
        axes[1].set_ylabel("Eval Loss")

    # Đồ thị 3: Evaluation BLEU Score
    if not df_eval.empty and "eval_bleu" in df_eval.columns:
        x_col = "epoch" if "epoch" in df_eval.columns and df_eval["epoch"].nunique() > 1 else "step"
        sns.lineplot(ax=axes[2], data=df_eval, x=x_col, y="eval_bleu", color="green", marker="s", linewidth=2)
        axes[2].set_title(f"BLEU Score theo {x_col.capitalize()}", fontsize=14)
        axes[2].set_xlabel(x_col.capitalize())
        axes[2].set_ylabel("SacreBLEU Score")

    plt.tight_layout()
    plot_path = "outputs/training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Đã lưu Biểu đồ Quá trình Huấn luyện tại: {plot_path}")
    
    # Hiển thị nếu chạy trong notebook (hoặc môi trường có màn hình)
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
