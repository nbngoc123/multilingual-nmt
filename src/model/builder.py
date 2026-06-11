import logging
# pyrefly: ignore [missing-import]
from transformers import AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

def load_mbart_model(model_name: str, cache_dir: str = "./cache"):
    """
    Hàm khởi tạo Neural Network cho mô hình mBART50.
    
    Args:
        model_name: Tên của mô hình trên HuggingFace (vd: facebook/mbart-large-50-many-to-many-mmt)
        cache_dir: Thư mục lưu trữ bộ nhớ đệm
        
    Returns:
        Đối tượng mBART Model
    """
    logger.info(f"Đang tải Model từ: {model_name}")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        return model
    except Exception as e:
        logger.error(f"Lỗi khi tải Model: {str(e)}")
        raise
