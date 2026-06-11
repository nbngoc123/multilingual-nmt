import os
import sys
import logging

# Thêm thư mục gốc vào đường dẫn để import được thư mục src/
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.config import load_config
from src.model.tokenizer import load_mbart_tokenizer
from src.model.builder import load_mbart_model
from src.inference.translator import MBartTranslator

def main():
    print("\n" + "="*60)
    print("🚀 BỘ DỊCH THUẬT ĐA NGÔN NGỮ mBART50 🚀")
    print("="*60)
    
    # 1. Tải cấu hình
    model_config = load_config("configs/model_config.yaml")
    inference_config = load_config("configs/inference_config.yaml")
    
    # 2. Tải mô hình và Tokenizer
    model_name = model_config["model_name"]
    print(f"\n[1] Đang tải mô hình từ: {model_name}...")
    
    # Ở bài toán thực tế, nếu bạn đã train xong thì sửa chỗ này thành đường dẫn
    # tới mô hình của bạn. Ví dụ: "./saved_models/mbart50-balanced-final"
    tokenizer = load_mbart_tokenizer(model_name)
    model = load_mbart_model(model_name)
    
    # 3. Khởi tạo Trợ lý Dịch (Translator)
    print("\n[2] Khởi tạo Translator thành công!")
    translator = MBartTranslator(model, tokenizer)
    
    # 4. Kịch bản So sánh Chiến lược Sampling (Giống bài giảng AI VIỆT NAM)
    src_text = "I’m so tired today."
    print("\n" + "-"*60)
    print(f"Câu Tiếng Anh gốc   : {src_text}")
    print("-"*60)
    
    # Thử nghiệm 3 chế độ lấy mẫu token khác nhau
    strategies = ["top_p", "top_k", "both"]
    
    for strategy in strategies:
        # Cập nhật tạm thời chế độ vào biến config
        inference_config["strategy"] = strategy
        
        # Dịch
        result = translator.translate(
            text=src_text,
            src_lang="en",
            tgt_lang="vi",
            inference_config=inference_config
        )
        
        print(f"[{strategy.upper().ljust(5)}] -> {result}")
        
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
