import os
import sys
import pandas as pd
from tqdm import tqdm

# Thêm thư mục gốc vào đường dẫn để import
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.config import load_config
from src.model.tokenizer import load_mbart_tokenizer
from src.model.builder import load_mbart_model
from src.inference.translator import MBartTranslator
from src.data.loader import load_multilingual_dataset

def main():
    print("="*60)
    print("🚀 BẮT ĐẦU DỊCH HÀNG LOẠT TRÊN TẬP TEST 🚀")
    print("="*60)
    
    # 1. Tải cấu hình
    data_config = load_config("configs/data_config.yaml")
    inference_config = load_config("configs/inference_config.yaml")
    
    # Thư mục chứa mô hình đã huấn luyện
    model_path = "./saved_models/mbart50-balanced-final"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}. Vui lòng kiểm tra lại quá trình huấn luyện.")
    
    # 2. Khởi tạo mô hình và Translator
    print(f"\n[1] Đang tải mô hình từ: {model_path}")
    tokenizer = load_mbart_tokenizer(model_path)
    model = load_mbart_model(model_path)
    translator = MBartTranslator(model, tokenizer)
    
    # 3. Tải bộ dữ liệu Test
    print("\n[2] Đang tải tập dữ liệu Test...")
    dataset = load_multilingual_dataset(
        lang_pairs=data_config["lang_pairs"],
        max_samples=10, # Bỏ qua train vì chỉ cần lấy test
        val_samples=10,
        test_samples=data_config.get("test_samples", 100),
        cache_dir="./data/raw"
    )
    test_data = dataset["test"]
    print(f"Tổng số mẫu Test sẽ dịch: {len(test_data)}")
    
    # 4. Chạy Inference qua vòng lặp
    results = []
    
    print("\n[3] Bắt đầu dịch...")
    for example in tqdm(test_data, desc="Tiến độ dịch"):
        src_text = example["src"]
        tgt_text = example["tgt"]
        pair = example["pair"]
        
        # Lấy ngôn ngữ nguồn và đích từ pair (vd: "en-vi" -> src="en", tgt="vi")
        src_lang, tgt_lang = pair.split("-")
        
        # Dịch
        try:
            pred_text = translator.translate(
                text=src_text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                inference_config=inference_config
            )
        except Exception as e:
            print(f"Lỗi dịch với câu: '{src_text}'. Lỗi: {str(e)}")
            pred_text = ""
            
        results.append({
            "pair": pair,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "source_text": src_text,
            "target_true": tgt_text,
            "prediction": pred_text
        })
        
    # 5. Lưu ra file CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/test_inference_results.csv"
    df = pd.DataFrame(results)
    
    # Dùng utf-8-sig để Excel đọc được Tiếng Việt không bị lỗi font
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*60)
    print(f"✅ HOÀN TẤT! Đã lưu kết quả của {len(results)} câu vào file: {csv_path}")
    print("="*60)

if __name__ == "__main__":
    main()
