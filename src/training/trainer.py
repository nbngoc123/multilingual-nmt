# pyrefly: ignore [missing-import]
from transformers import Seq2SeqTrainer
import torch
import logging

from src.model.loss import BalancedLoss

logger = logging.getLogger(__name__)

class BalancedLossSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Trainer Tùy Chỉnh: Kế thừa toàn bộ sức mạnh của Seq2SeqTrainer từ Hugging Face
    nhưng ghi đè hàm tính Loss để phạt nặng các mẫu ngôn ngữ hiếm.
    """
    def __init__(self, pair_counts: dict, smoothing_factor: float = 0.5, *args, **kwargs):
        # Khởi tạo Trainer gốc
        super().__init__(*args, **kwargs)
        
        # Khởi tạo hàm tính Loss tùy chỉnh của chúng ta
        self.pair_counts = pair_counts
        self.custom_loss_fct = BalancedLoss(
            pair_counts=pair_counts, 
            smoothing_factor=smoothing_factor
        )
        logger.info(f"Đã khởi tạo BalancedLossSeq2SeqTrainer thành công!")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Ghi đè hàm tính Loss mặc định.
        """
        pairs = inputs.pop("pair", None)
        
        # 1. Forward Pass: Đưa toàn bộ inputs còn lại (toàn số) vào mô hình
        outputs = model(**inputs)
        
        # Nếu không có cột pair (thường xảy ra ở bước Evaluation), dùng loss mặc định
        if pairs is None:
            loss = outputs.get("loss")
            return (loss, outputs) if return_outputs else loss

        # 2. Đẩy dự đoán, đáp án và danh sách "en-vi" vào cái Cân (BalancedLoss) để tính phạt
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss = self.custom_loss_fct(logits, labels, pairs)
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Ghi đè bước Evaluation để loại bỏ triệt để cột 'pair' trước khi chui vào model.generate()
        """
        _ = inputs.pop("pair", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
