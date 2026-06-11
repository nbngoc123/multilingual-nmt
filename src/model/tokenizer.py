import logging
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def load_mbart_tokenizer(model_name: str, cache_dir: str = "./cache"):
    """
    Hàm khởi tạo Tokenizer cho mô hình mBART50.
    
    Args:
        model_name: Tên của mô hình trên HuggingFace (vd: facebook/mbart-large-50-many-to-many-mmt)
        cache_dir: Thư mục lưu trữ bộ nhớ đệm
        
    Returns:
        Đối tượng mBART Tokenizer
    """
    logger.info(f"Đang tải Tokenizer từ: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=False,
            cache_dir=cache_dir
        )
        return tokenizer
    except Exception as e:
        logger.error(f"Lỗi khi tải Tokenizer: {str(e)}")
        raise
