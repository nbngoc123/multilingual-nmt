# pyrefly: ignore [missing-import]
from transformers import PreTrainedTokenizer

LANG_TO_MBART = {
    "en": "en_XX",
    "vi": "vi_VN",
    "fr": "fr_XX",
    "de": "de_DE",
    "zh": "zh_CN",
    "ja": "ja_XX",
}

class MultilingualPreprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def preprocess_function(self, examples):
        input_ids, attention_mask, labels = [], [], []
        
        for src, tgt, pair in zip(examples["src"], examples["tgt"], examples["pair"]):
            src_lang, tgt_lang = pair.split("-")
            
            # Cài đặt ngôn ngữ cho mBART Tokenizer
            self.tokenizer.src_lang = LANG_TO_MBART.get(src_lang, "en_XX")
            self.tokenizer.tgt_lang = LANG_TO_MBART.get(tgt_lang, "en_XX")
            
            # Tokenize câu nguồn (input)
            enc = self.tokenizer(
                src,
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            # Tokenize câu đích (labels)
            dec = self.tokenizer(
                text_target=tgt,
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            # Thay đổi padding token id thành -100
            pad_id = self.tokenizer.pad_token_id
            sample_labels = [-100 if t == pad_id else t for t in dec["input_ids"]]
            
            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            labels.append(sample_labels)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pair": examples["pair"]
        }
