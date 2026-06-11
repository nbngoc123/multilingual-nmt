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
        # Bước cực kỳ quan trọng:
        # Nhấc cột 'pair' ra khỏi dict `inputs` trước khi đưa vào Model.
        # Nếu để quên chữ 'en-vi' bay vào mô hình Neural Network, nó sẽ crash ngay lập tức!
        pairs = inputs.pop("pair", None)
        
        if pairs is None:
            raise ValueError("Lỗi nghiêm trọng: Không tìm thấy cột 'pair' trong inputs. Vui lòng kiểm tra lại DataCollator!")

        # 1. Forward Pass: Đưa toàn bộ inputs còn lại (toàn số) vào mô hình
        outputs = model(**inputs)
        
        # Lấy Logits (dự đoán của mạng) và Labels (đáp án đúng)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        # 2. Đẩy dự đoán, đáp án và danh sách "en-vi" vào cái Cân (BalancedLoss) để tính phạt
        loss = self.custom_loss_fct(logits, labels, pairs)
        
        # Trả về kết quả theo chuẩn format của HuggingFace
        return (loss, outputs) if return_outputs else loss
