import json
import os

def save_evaluation_report(metrics: dict, output_dir: str = "./reports", filename: str = "eval_results.json"):
    """
    Lưu kết quả điểm số (ví dụ: BLEU score) ra file JSON
    để tiện lợi cho việc báo cáo (report) và phân tích sau này.
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Ghép tên file
    filepath = os.path.join(output_dir, filename)
    
    # Ghi dữ liệu dạng JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
