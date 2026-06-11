import numpy as np
# pyrefly: ignore [missing-import]
import evaluate

def get_compute_metrics(tokenizer):
    """
    Hàm Factory trả về hàm compute_metrics dành cho Trainer.
    Sử dụng sacrebleu để chấm điểm chất lượng dịch thuật.
    """
    # Tải thư viện tính điểm BLEU chuẩn quốc tế (SacreBLEU)
    metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Mô hình Seq2Seq đôi khi trả về tuple thay vì numpy array thuần
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Giải mã (Decode) các mảng số ID thành văn bản chữ con người đọc được
        # skip_special_tokens=True để bỏ qua mấy cái như </s>, <pad>
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Nhãn (Labels) của chúng ta đang chứa số -100 ở những chỗ padding.
        # Tokenizer không hiểu số -100 này là gì và sẽ văng lỗi. 
        # Cần thay thế -100 lại thành pad_token_id an toàn.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Xóa khoảng trắng thừa ở đầu/cuối câu
        decoded_preds = [pred.strip() for pred in decoded_preds]
        # Thư viện sacrebleu yêu cầu đáp án (references) phải là một danh sách các danh sách
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        # Ném câu dự đoán và câu đáp án vào cho metric chấm điểm
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Trả về duy nhất điểm BLEU
        return {"bleu": result["score"]}
        
    return compute_metrics
