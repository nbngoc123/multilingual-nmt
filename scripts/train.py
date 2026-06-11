import os
import sys
from collections import Counter

# Thêm thư mục gốc vào đường dẫn để import được src/
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.loader import load_multilingual_dataset
from src.data.preprocessor import MultilingualPreprocessor
from src.data.collator import custom_collate_fn
from src.model.tokenizer import load_mbart_tokenizer
from src.model.builder import load_mbart_model
from src.training.arguments import get_training_args
from src.training.callbacks import get_callbacks
from src.training.trainer import BalancedLossSeq2SeqTrainer
from src.evaluation.metrics import get_compute_metrics

def main():
    # 1. Môi trường cơ sở
    logger = setup_logger("train_script")
    set_seed(42)
    logger.info("Bắt đầu kịch bản huấn luyện NMT Đa Ngôn Ngữ với Balanced Loss!")

    # 2. Tải cấu hình
    data_config = load_config("configs/data_config.yaml")
    train_config = load_config("configs/train_config.yaml")
    model_config = load_config("configs/model_config.yaml")

    # 3. Tải Data
    logger.info("Đang tải dữ liệu OPUS-100...")
    dataset = load_multilingual_dataset(
        lang_pairs=data_config["lang_pairs"],
        max_samples=data_config["max_samples"],
        val_samples=data_config["val_samples"],
        test_samples=data_config["test_samples"]
    )
    
    # Tính tần suất ngôn ngữ từ tập Train để chia trọng số phạt
    pair_counts = dict(Counter(dataset["train"]["pair"]))
    logger.info(f"Tần suất cặp ngôn ngữ (phục vụ Balanced Loss): {pair_counts}")

    # 4. Tải Model & Tokenizer
    tokenizer = load_mbart_tokenizer(model_config["model_name"])
    model = load_mbart_model(model_config["model_name"])

    # 5. Tiền xử lý dữ liệu (Tokenization)
    logger.info("Tiến hành Tokenize toàn bộ tập dữ liệu...")
    preprocessor = MultilingualPreprocessor(tokenizer, max_length=data_config["max_length"])
    
    tokenized_datasets = dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=["src", "tgt"] # Xóa cột văn bản thô, nhưng GIỮ LẠI cột 'pair'
    )
    
    # 6. Cấu hình Training
    training_args = get_training_args(train_config, output_dir="./saved_models/mbart50-balanced")
    callbacks = get_callbacks()

    # 7. Khởi tạo BalancedLossSeq2SeqTrainer
    logger.info("Gắn kết Model, Data, Loss vào Trainer...")
    trainer = BalancedLossSeq2SeqTrainer(
        pair_counts=pair_counts,
        smoothing_factor=0.5,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer, # transformers bản mới đổi tên tham số này
        data_collator=custom_collate_fn, 
        compute_metrics=get_compute_metrics(tokenizer),
        callbacks=callbacks
    )

    # 8. Bắt đầu Vòng lặp Huấn luyện (Training Loop)
    logger.info("🚀 Bắt đầu HUẤN LUYỆN!")
    trainer.train()

    # 9. Lưu trữ sau khi chạy xong
    logger.info("Đang lưu mô hình hoàn chỉnh...")
    trainer.save_model("./saved_models/mbart50-balanced-final")
    tokenizer.save_pretrained("./saved_models/mbart50-balanced-final")
    logger.info("Tuyệt vời! Đã hoàn tất huấn luyện!")

if __name__ == "__main__":
    main()
