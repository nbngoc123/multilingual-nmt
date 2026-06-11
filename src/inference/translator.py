import torch
import logging

from src.data.preprocessor import LANG_TO_MBART
from src.inference.sampler import get_sampling_kwargs

logger = logging.getLogger(__name__)

class MBartTranslator:
    """
    Class Đóng Gói (Wrapper) dùng để dịch văn bản cực nhanh.
    Phù hợp để gắn vào Web App, API hoặc Bot Chat sau này.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Tự động chọn GPU nếu có, không thì chạy CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Đưa mô hình về chế độ Suy luận (Không tính toán Gradient)
        self.model.eval()
        
    def translate(self, text: str, src_lang: str, tgt_lang: str, inference_config: dict) -> str:
        """
        Thực hiện dịch thuật với tham số Top-k/Top-p từ inference_config.
        """
        # 1. Trích xuất đúng mã ngôn ngữ mBART (vd: "en" -> "en_XX")
        src_code = LANG_TO_MBART.get(src_lang)
        tgt_code = LANG_TO_MBART.get(tgt_lang)
        
        if not src_code or not tgt_code:
            raise ValueError(f"Ngôn ngữ {src_lang} hoặc {tgt_lang} chưa được hỗ trợ trong danh sách.")
            
        # 2. Gán cờ ngôn ngữ nguồn cho Tokenizer
        self.tokenizer.src_lang = src_code
        
        # 3. Mã hóa câu đầu vào thành Tensor
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # 4. Tính toán thông số Sampling từ cấu hình YAML
        strategy = inference_config.get("strategy", "both")
        sampling_kwargs = get_sampling_kwargs(strategy, inference_config)
        
        # 5. Sinh chuỗi văn bản (Dịch)
        # Sử dụng torch.no_grad() cực kỳ quan trọng để tiết kiệm RAM
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                **sampling_kwargs
            )
            
        # 6. Giải mã kết quả (Decode)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result
