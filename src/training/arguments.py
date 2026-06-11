# pyrefly: ignore [missing-import]
from transformers import Seq2SeqTrainingArguments
import os

def get_training_args(train_config: dict, output_dir: str = "./saved_models"):
    """
    Khởi tạo các tham số huấn luyện (Training Arguments) cho Seq2SeqTrainer.
    Các giá trị mặc định được lấy TOÀN BỘ từ train_config.yaml.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy=train_config.get("eval_strategy", "epoch"),
        save_strategy=train_config.get("save_strategy", "epoch"),
        learning_rate=train_config.get("lr", 5e-5),
        per_device_train_batch_size=train_config.get("batch_size", 8),
        per_device_eval_batch_size=train_config.get("batch_size", 8),
        weight_decay=train_config.get("weight_decay", 0.01),
        save_total_limit=train_config.get("save_total_limit", 3),
        num_train_epochs=train_config.get("num_epochs", 3),
        predict_with_generate=True,         
        fp16=train_config.get("fp16", True),
        push_to_hub=False,
        report_to="none",
        logging_dir="./logs",
        logging_steps=train_config.get("logging_steps", 10),
        warmup_steps=train_config.get("warmup_steps", 500),
        load_best_model_at_end=True,                   # Yêu cầu bắt buộc của EarlyStopping
        metric_for_best_model="eval_bleu",             # Dựa vào điểm BLEU để dừng
        greater_is_better=True,                        # BLEU càng cao càng tốt
        remove_unused_columns=False                    # QUAN TRỌNG: Ngăn Trainer xóa cột 'pair'
    )
