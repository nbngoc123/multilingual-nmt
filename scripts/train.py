import sys
import os

# Đảm bảo python hiểu thư mục gốc là multilingual-nmt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
# pyrefly: ignore [missing-import]
from transformers import Seq2SeqTrainingArguments
from collections import Counter

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.loader import load_multilingual_dataset
from src.data.preprocessor import MultilingualPreprocessor
from src.data.collator import custom_collate_fn
from src.model.tokenizer import load_mbart_tokenizer
from src.model.builder import load_mbart_model
from src.training.trainer import BalancedLossSeq2SeqTrainer

def main():
    # 1. Khởi tạo môi trường
    logger = setup_logger("train_script")
    set_seed(42)
    logger.info("Đã cố định Random Seed (42). Bắt đầu quá trình chuẩn bị huấn luyện...")
    
    # 2. Tải cấu hình
    data_config = load_config("configs/data_config.yaml")
    train_config = load_config("configs/train_config.yaml")
    model_config = load_config("configs/model_config.yaml")
    model_name = model_config["model_name"]
    
    logger.info(f"Cấu hình Train: Batch size={train_config['batch_size']}, LR={train_config['lr']}")
    
    # 3. Chuẩn bị Dữ liệu
    dataset = load_multilingual_dataset(
        lang_pairs=data_config["lang_pairs"],
        max_samples=data_config["max_samples"],
        val_samples=data_config["val_samples"],
        test_samples=data_config["test_samples"]
    )
    
    # Lấy phân phối pair để làm trọng số cho BalancedLoss
    pair_counts = dict(Counter(dataset["train"]["pair"]))
    logger.info(f"Số lượng mẫu huấn luyện của từng cặp ngôn ngữ: {pair_counts}")
    
    # 4. Tải Tokenizer và Model
    tokenizer = load_mbart_tokenizer(model_name)
    model = load_mbart_model(model_name)
    
    # 5. Tiền xử lý dữ liệu (Tokenization)
    logger.info("Bắt đầu mã hóa (Tokenize) dữ liệu...")
    preprocessor = MultilingualPreprocessor(tokenizer, max_length=data_config["max_length"])
    
    tokenized_datasets = dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=dataset["train"].column_names # Xóa các cột text, chỉ giữ input_ids, mask, labels và pair
    )
    logger.info("Hoàn thành mã hóa dữ liệu.")
    
    # 6. Thiết lập tham số huấn luyện
    # Sử dụng fp16 (mixed precision) nếu máy tính có card đồ họa (GPU) để tránh tràn bộ nhớ VRAM
    use_fp16 = torch.cuda.is_available()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./saved_models/mbart_finetuned",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=train_config["lr"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        num_train_epochs=train_config["num_epochs"],
        warmup_steps=train_config["warmup_steps"],
        fp16=use_fp16,
        logging_dir="./logs",
        logging_steps=10,
        predict_with_generate=True, # Cho phép mô hình sinh ra từ mới lúc Eval
        seed=42,
        report_to="none" # Tắt report lên WandB để tránh bị hỏi đăng nhập ở phiên chạy thử
    )
    
    # 7. Khởi tạo Custom Trainer
    logger.info("Đang khởi tạo BalancedLossSeq2SeqTrainer...")
    trainer = BalancedLossSeq2SeqTrainer(
        pair_counts=pair_counts,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=custom_collate_fn
    )
    
    # 8. Bắt đầu huấn luyện
    logger.info("🚀 Bắt đầu quá trình huấn luyện (Training)...")
    trainer.train()
    
    # 9. Lưu mô hình
    logger.info("Đã huấn luyện xong! Đang tiến hành lưu mô hình...")
    trainer.save_model("./saved_models/mbart_final")
    tokenizer.save_pretrained("./saved_models/mbart_final")
    logger.info("✨ Toàn bộ quy trình hoàn tất thành công!")

if __name__ == "__main__":
    main()
