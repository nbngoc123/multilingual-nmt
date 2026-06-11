import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu

def main():
    csv_path = "outputs/test_inference_results.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file {csv_path}. Bạn cần chạy batch_infer.py để sinh dữ liệu trước nhé.")
        return

    print("Đang đọc dữ liệu từ CSV và tính toán...")
    # Đọc dữ liệu
    df = pd.DataFrame(pd.read_csv(csv_path))
    
    # 1. Tính điểm BLEU cho từng cặp ngôn ngữ
    pairs = df["pair"].unique()
    bleu_scores = {}
    
    for pair in pairs:
        subset = df[df["pair"] == pair]
        
        # Xử lý các giá trị NaN (nếu mô hình không dịch được)
        preds = subset["prediction"].fillna("").tolist()
        refs_list = subset["target_true"].fillna("").tolist()
        
        # SacreBLEU yêu cầu danh sách tham chiếu có dạng: [[ref1_của_câu1, ref1_của_câu2...]]
        refs = [refs_list]
        
        score = sacrebleu.corpus_bleu(preds, refs)
        bleu_scores[pair] = score.score
        
    # Chuyển thành DataFrame để vẽ
    bleu_df = pd.DataFrame(list(bleu_scores.items()), columns=["Language Pair", "BLEU Score"])
    bleu_df = bleu_df.sort_values(by="BLEU Score", ascending=False)
    
    # 2. Bắt đầu vẽ biểu đồ
    os.makedirs("outputs", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # ==========================================
    # BIỂU ĐỒ 1: SO SÁNH ĐIỂM BLEU
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Vẽ Barplot
    ax = sns.barplot(x="Language Pair", y="BLEU Score", data=bleu_df, hue="Language Pair", palette="viridis", legend=False)
    plt.title("Đánh giá độ chính xác (SacreBLEU Score) trên từng Cặp Ngôn Ngữ", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Cặp ngôn ngữ", fontsize=12)
    plt.ylabel("SacreBLEU Score", fontsize=12)
    
    # Gắn số điểm trực tiếp lên từng cột
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
                    
    # Lưu biểu đồ
    plot_path1 = "outputs/bleu_scores_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path1, dpi=300)
    print(f"✅ Đã lưu biểu đồ Điểm BLEU tại: {plot_path1}")
    
    # ==========================================
    # BIỂU ĐỒ 2: PHÂN PHỐI ĐỘ DÀI CÂU DỊCH
    # ==========================================
    # Tính số từ (word count) của câu gốc và câu dự đoán
    df["src_word_count"] = df["source_text"].apply(lambda x: len(str(x).split()))
    df["pred_word_count"] = df["prediction"].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="src_word_count", y="pred_word_count", hue="pair", alpha=0.7, s=80)
    
    # Vẽ đường chéo y=x để làm mốc (Câu gốc và câu dịch có độ dài bằng nhau)
    max_len = max(df["src_word_count"].max(), df["pred_word_count"].max())
    plt.plot([0, max_len], [0, max_len], 'r--', label='Độ dài tương đương (y=x)', linewidth=2)
    
    plt.title("So sánh Độ dài Câu Nguồn và Câu Dịch", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Số từ của Câu Gốc", fontsize=12)
    plt.ylabel("Số từ của Câu Dịch", fontsize=12)
    plt.legend(title="Cặp ngôn ngữ")
    
    # Lưu biểu đồ
    plot_path2 = "outputs/length_distribution_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=300)
    print(f"✅ Đã lưu biểu đồ Phân phối độ dài tại: {plot_path2}")

if __name__ == "__main__":
    main()
