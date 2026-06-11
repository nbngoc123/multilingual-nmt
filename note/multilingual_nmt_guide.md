# Multilingual Neural Machine Translation — Hướng dẫn triển khai chi tiết

> Fine-tune **mBART50** trên OPUS-100 để dịch many-to-many giữa nhiều ngôn ngữ,
> áp dụng Balanced Loss và Top-k/Top-p sampling.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Chuẩn bị dữ liệu](#2-chuẩn-bị-dữ-liệu)
3. [Tokenization với mBART50](#3-tokenization-với-mbart50)
4. [Xây dựng model](#4-xây-dựng-model)
5. [Balanced Loss](#5-balanced-loss)
6. [Huấn luyện](#6-huấn-luyện)
7. [Đánh giá — SacreBLEU](#7-đánh-giá--sacrebleu)
8. [Inference — Sampling strategies](#8-inference--sampling-strategies)
9. [Luồng dữ liệu end-to-end](#9-luồng-dữ-liệu-end-to-end)
10. [Checklist triển khai](#10-checklist-triển-khai)

---

## 1. Tổng quan kiến trúc

### 1.1 Tại sao dùng mBART50?

| Tiêu chí | BART | T5 | **mBART50** |
|---|---|---|---|
| Đơn ngữ | ✅ Tiếng Anh | ✅ Tiếng Anh | ❌ |
| Đa ngôn ngữ | ❌ | Có biến thể mT5 | ✅ ~50 ngôn ngữ |
| Language code | ❌ | Prefix text | ✅ Token đặc biệt |
| Many-to-many | ❌ | Có thể | ✅ Thiết kế sẵn |
| Kiến trúc | Encoder–Decoder | Encoder–Decoder | Encoder–Decoder |

mBART50 (`facebook/mbart-large-50-many-to-many-mmt`) là lựa chọn tối ưu vì:

- Đã được pre-train trên 50 ngôn ngữ bằng mục tiêu **denoising** — model hiểu cấu trúc ngôn ngữ tự nhiên sâu hơn
- Dùng **language code token** (`en_XX`, `vi_VN`, `fr_XX`, `de_DE`) để điều khiển hướng dịch — một model duy nhất xử lý tất cả cặp
- Tokenizer **SentencePiece** với shared vocabulary 250.000 tokens giữa tất cả ngôn ngữ

### 1.2 Kiến trúc Encoder–Decoder

```
Câu nguồn (src)                Câu đích (tgt)
     │                               │
     ▼                               ▼
[Tokenize + src_lang code]    [Tokenize → labels]
     │
     ▼
┌─────────────────────┐
│   Transformer       │   ← 12 encoder layers
│   Encoder           │     hidden_size = 1024
│   (Bidirectional)   │     attention_heads = 16
└────────┬────────────┘
         │ context vectors
         ▼
┌─────────────────────┐
│   Transformer       │   ← 12 decoder layers
│   Decoder           │     cross-attention lên encoder
│   (Autoregressive)  │     forced_bos = tgt_lang code
└────────┬────────────┘
         │
         ▼
    Token dự đoán → Câu đích
```

Điểm khác biệt so với BART thông thường: decoder bắt đầu bằng **language code token** của ngôn ngữ đích, ép model sinh đúng ngôn ngữ mong muốn.

---

## 2. Chuẩn bị dữ liệu

### 2.1 Dataset — OPUS-100

OPUS-100 là tập hợp dữ liệu song ngữ từ nhiều nguồn (web, phụ đề, tài liệu), được tổ chức theo config name theo dạng `{src}-{tgt}`.

**Các config dùng trong project:**

| Config | Src | Tgt | Train size | Ghi chú |
|---|---|---|---|---|
| `en-vi` | English | Vietnamese | ~500K | Nhiều nhất |
| `en-fr` | English | French | ~1M | Rất nhiều |
| `de-en` | German | English | ~400K | Ngược hướng |

**Giới hạn `MAX_SAMPLES_PER_PAIR = 50000`** khi experiment để kiểm soát thời gian train. Khi deploy thực tế có thể tăng lên toàn bộ dataset.

### 2.2 Schema chuẩn hóa

Mỗi sample sau khi xử lý có 3 trường:

```python
{
    "pair": "en-vi",        # dùng để tính balanced loss
    "src":  "I love you",   # câu nguồn
    "tgt":  "Tôi yêu bạn",  # câu đích
}
```

Trường `pair` **bắt buộc phải giữ** xuyên suốt pipeline — nó là key để tra cứu trọng số balanced loss khi training.

### 2.3 Phân tích mất cân bằng dữ liệu

Trước khi train, luôn kiểm tra phân phối:

```python
from collections import Counter
pair_counts = Counter(ds["train"]["pair"])
# {'en-vi': 50000, 'en-fr': 50000, 'de-en': 50000}
# → cân bằng nhờ MAX_SAMPLES_PER_PAIR
# Nhưng trong thực tế không giới hạn, en-fr >> de-en
```

Nếu mất cân bằng nặng (tỷ lệ > 10:1), Balanced Loss là **bắt buộc** — không phải tùy chọn.

---

## 3. Tokenization với mBART50

### 3.1 Language code mapping

mBART50 dùng token đặc biệt để nhận biết ngôn ngữ. Mapping cần chính xác:

```python
LANG_TO_MBART = {
    "en": "en_XX",   # English — XX vì không có region cụ thể
    "vi": "vi_VN",   # Vietnamese — Vietnam
    "fr": "fr_XX",   # French
    "de": "de_DE",   # German — Germany
    "zh": "zh_CN",   # Chinese Simplified
    "ja": "ja_XX",   # Japanese
}
```

Nếu dùng sai language code, model sẽ sinh ra ngôn ngữ sai hoặc văn bản lộn xộn — đây là lỗi phổ biến nhất khi bắt đầu.

### 3.2 Quy trình tokenize một sample

```
Input:  src="I love you", tgt="Tôi yêu bạn", pair="en-vi"

Bước 1: Đặt src_lang và tgt_lang
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "vi_VN"

Bước 2: Tokenize câu nguồn → input cho Encoder
    enc = tokenizer(src, padding="max_length",
                    truncation=True, max_length=128)
    enc["input_ids"]      → [en_XX_id, 100, 231, 89, ..., pad, pad]
    enc["attention_mask"] → [1, 1, 1, 1, ..., 0, 0]

Bước 3: Tokenize câu đích → labels cho Decoder
    dec = tokenizer(tgt, padding="max_length",
                    truncation=True, max_length=128)
    dec["input_ids"] → [vi_VN_id, 500, 312, 77, ..., pad, pad]

Bước 4: Tạo labels — thay PAD bằng -100
    labels = [-100 if t == pad_id else t
              for t in dec["input_ids"]]
    # -100 → CrossEntropyLoss tự động bỏ qua khi tính loss
```

### 3.3 Tại sao gán -100 cho padding?

`CrossEntropyLoss(ignore_index=-100)` — đây là convention của PyTorch. Nếu không gán -100, loss sẽ tính cả trên padding tokens, làm model học sai mục tiêu và loss không phản ánh đúng chất lượng dịch.

### 3.4 Output tensor sau tokenize

Với `batch_size=B`, `max_length=T`, `vocab_size=V`:

```
input_ids:       (B, T)     — index vào embedding table
attention_mask:  (B, T)     — 1=token thật, 0=padding
labels:          (B, T)     — target cho decoder, -100 ở vị trí pad
logits (output): (B, T, V)  — xác suất phân phối trên vocab
```

---

## 4. Xây dựng model

### 4.1 Load mBART50

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/mbart-large-50-many-to-many-mmt"

# Tokenizer — dùng use_fast=False vì SentencePiece
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Model — ~610M parameters
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

**Lưu ý môi trường:**

- RAM tối thiểu: 16GB (load model + batch)
- GPU VRAM tối thiểu: 16GB (batch_size=8, max_len=128)
- Nếu chỉ có 8GB VRAM: dùng `gradient_checkpointing=True` và `batch_size=4`

### 4.2 Tính pair weights cho Balanced Loss

```python
import math
from collections import Counter

pair_counts = Counter(ds["train"]["pair"])
total = len(ds["train"])

# Công thức: weight = sqrt(total / count)
# → cặp ít dữ liệu có weight lớn hơn, nhưng không quá extreme
pair_weights = {
    pair: math.sqrt(total / count)
    for pair, count in pair_counts.items()
}

# Ví dụ với dữ liệu thực tế (không giới hạn MAX_SAMPLES):
# en-fr: 1M samples → weight = sqrt(1.5M / 1M) = 1.22
# de-en: 200K samples → weight = sqrt(1.5M / 200K) = 2.74
# → de-en được "khuếch đại" ~2.25x so với en-fr
```

---

## 5. Balanced Loss

### 5.1 Vấn đề cần giải quyết

Khi gộp nhiều language pair vào cùng một batch:

- Cặp nhiều dữ liệu (en-fr, en-vi) xuất hiện nhiều → gradient lớn → model tối ưu tốt cho chúng
- Cặp ít dữ liệu (de-en, zh-vi) xuất hiện ít → gradient nhỏ → model "quên" chúng

Kết quả: BLEU của cặp phổ biến cao, cặp hiếm gần như không cải thiện.

### 5.2 Công thức

$$\mathcal{L}_{\text{balanced}} = \frac{\sum_{i=1}^{B} w(\text{pair}_i) \cdot L_i}{\sum_{i=1}^{B} w(\text{pair}_i)}$$

Trong đó:
- $L_i$ = loss trung bình trên các token hợp lệ của sample $i$ (bỏ qua -100)
- $w(\text{pair}_i)$ = trọng số của cặp ngôn ngữ mà sample $i$ thuộc về
- Mẫu số $\sum w$ — giữ loss ở cùng thang đo, tránh phình lên khi weight lớn

### 5.3 Triển khai step-by-step

```python
# Bước 1: Lấy danh sách pair và bỏ ra khỏi inputs
# (model không nhận tham số "pair", chỉ dùng để tra weight)
pairs = inputs.pop("pair")

# Bước 2: Forward pass
outputs = model(**inputs)
logits = outputs.logits          # (B, T, V)
labels = inputs["labels"]        # (B, T)

# Bước 3: Tính loss per token, không reduction
loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
loss_flat = loss_fct(
    logits.view(-1, V),   # (B*T, V)
    labels.view(-1)       # (B*T,)
)

# Bước 4: Reshape về (B, T)
loss_per_token = loss_flat.view(B, T)

# Bước 5: Tính loss per sample (mean trên token hợp lệ)
mask = (labels != -100).float()                   # (B, T)
loss_per_sample = (loss_per_token * mask).sum(1)  # (B,)
               / mask.sum(1).clamp(min=1)         # tránh chia 0

# Bước 6: Áp dụng pair weights
weights = torch.tensor(
    [pair_weights.get(p, 1.0) for p in pairs],
    device=loss_per_sample.device
)                                                  # (B,)

# Bước 7: Weighted mean
loss = (weights * loss_per_sample).sum() / weights.sum()
```

### 5.4 Tại sao dùng `sqrt(total/count)` thay vì `total/count`?

- `total/count` (linear inverse): cặp có 100 samples được weight 1000x so với cặp 100K → quá extreme, gradient bị dominate bởi cặp hiếm
- `sqrt(total/count)` (square root): làm mượt hơn, cặp hiếm vẫn được ưu tiên nhưng không quá mức
- Trong thực tế, có thể thử `(total/count)^α` với `α ∈ (0.5, 0.7)` để tune

---

## 6. Huấn luyện

### 6.1 Training arguments quan trọng

```python
Seq2SeqTrainingArguments(
    output_dir="./outputs/checkpoints",
    num_train_epochs=2,               # 2-3 epoch đủ với mBART50 pre-trained
    per_device_train_batch_size=8,    # giảm xuống 4 nếu OOM
    per_device_eval_batch_size=8,
    learning_rate=5e-5,               # standard fine-tune LR cho encoder-decoder
    warmup_steps=500,                 # warm up ~5% tổng steps
    eval_strategy="epoch",            # eval sau mỗi epoch
    save_strategy="epoch",
    predict_with_generate=True,       # QUAN TRỌNG: eval bằng generate, không argmax
    remove_unused_columns=False,      # QUAN TRỌNG: giữ cột "pair" trong batch
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    fp16=True,                        # mixed precision, tiết kiệm VRAM
)
```

**Hai tham số không được quên:**
- `predict_with_generate=True`: nếu để False, evaluation dùng argmax trên logits thay vì beam search → BLEU sai hoàn toàn
- `remove_unused_columns=False`: HuggingFace Trainer mặc định xóa các cột model không nhận → mất trường `pair`

### 6.2 Custom collate_fn

Trainer mặc định không biết xử lý trường `pair` (list of strings, không thể convert sang tensor). Cần custom:

```python
def collate_fn(examples):
    return {
        "input_ids":      torch.tensor([e["input_ids"]      for e in examples]),
        "attention_mask": torch.tensor([e["attention_mask"] for e in examples]),
        "labels":         torch.tensor([e["labels"]         for e in examples]),
        "pair":           [e["pair"] for e in examples],  # giữ dạng list[str]
    }
```

Trong `prediction_step`, cần pop `pair` ra trước khi gọi `super()`:

```python
def prediction_step(self, model, inputs, ...):
    inputs = {k: v for k, v in inputs.items() if k != "pair"}
    return super().prediction_step(model, inputs, ...)
```

### 6.3 Monitoring trong quá trình train

Theo dõi 3 chỉ số:

| Chỉ số | Mong đợi | Cảnh báo nếu |
|---|---|---|
| Train loss | Giảm đều từ ~3.5 xuống ~1.8 | Không giảm sau 2000 steps → LR sai |
| Val loss | Giảm nhẹ, ổn định | Tăng lên → overfitting |
| Val BLEU | Tăng từ ~18 lên ~22+ | Không tăng → `predict_with_generate` bị tắt |

---

## 7. Đánh giá — SacreBLEU

### 7.1 BLEU là gì?

BLEU (Bilingual Evaluation Understudy) đo mức độ khớp n-gram giữa câu dự đoán và câu tham chiếu:

$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Trong đó:
- $p_n$ = precision của n-gram (n=1,2,3,4)
- $w_n = 1/N$ = trọng số đều nhau
- BP = brevity penalty — phạt câu dịch quá ngắn

**Thang điểm thực tế:**

| BLEU | Đánh giá |
|---|---|
| < 10 | Rất kém, hầu như không dùng được |
| 10–19 | Kém, chỉ hiểu đại ý |
| 20–29 | Chấp nhận được, có thể dùng với hiệu đính |
| 30–40 | Tốt, chất lượng gần người dịch |
| > 40 | Rất tốt (hiếm gặp với many-to-many) |

### 7.2 Tại sao dùng SacreBLEU thay vì BLEU thông thường?

SacreBLEU chuẩn hóa tokenization trước khi tính → **có thể so sánh** kết quả giữa các paper và implementation khác nhau. BLEU thông thường phụ thuộc vào cách tokenize → không reproducible.

### 7.3 Quy trình compute_metrics

```python
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # preds có thể là tuple (sequences, scores) → lấy sequences
    if isinstance(preds, tuple):
        preds = preds[0]

    # Thay -100 bằng pad_token_id để decode không bị lỗi
    preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode về string
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]  # list of list!

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    return {"bleu": result["score"]}
```

**Điểm dễ nhầm:** `references` phải là `list[list[str]]` — mỗi sample có thể có nhiều câu tham chiếu, ở đây chỉ có 1 nhưng vẫn phải wrap thêm 1 cấp list.

---

## 8. Inference — Sampling strategies

### 8.1 Greedy vs Sampling

| Phương pháp | Mô tả | Ưu | Nhược |
|---|---|---|---|
| Greedy | Luôn chọn token xác suất cao nhất | Nhanh, nhất quán | Repetitive, thiếu đa dạng |
| Beam search | Duy trì k chuỗi tốt nhất song song | Cân bằng tốt | Chậm hơn, vẫn có thể repetitive |
| Top-k | Chọn ngẫu nhiên trong k token cao nhất | Đa dạng | k cố định, không thích nghi |
| Top-p (nucleus) | Chọn trong tập nhỏ nhất đạt tổng xác suất ≥ p | Tự thích nghi | Cần tune p |

### 8.2 Top-k sampling — cơ chế

```
Phân phối gốc:
  "the": 0.30, "a": 0.20, "an": 0.15, "this": 0.10,
  "that": 0.08, "my": 0.07, "our": 0.06, "some": 0.04

Top-k (k=3): giữ lại 3 token cao nhất
  → {"the": 0.30, "a": 0.20, "an": 0.15}

Normalize lại: tổng = 0.65
  → {"the": 0.462, "a": 0.308, "an": 0.231}

Sample từ phân phối đã normalize
```

**Nhược điểm:** Khi model rất chắc chắn (1 token có xác suất 0.99), k=50 vẫn ép lấy mẫu trong 50 token không cần thiết.

### 8.3 Top-p (nucleus) sampling — cơ chế

```
Phân phối gốc (sắp xếp giảm dần):
  0.40, 0.25, 0.15, 0.10, 0.05, 0.05

Top-p (p=0.80): tích lũy đến khi ≥ 0.80
  0.40 → tổng = 0.40 (chưa đủ)
  0.25 → tổng = 0.65 (chưa đủ)
  0.15 → tổng = 0.80 (đủ rồi!) → dừng

Nucleus = {token1, token2, token3}
Normalize và sample trong 3 token đó
```

**Ưu điểm:** Khi model chắc chắn → nucleus nhỏ → ít ngẫu nhiên. Khi model phân vân → nucleus lớn → đa dạng hơn. Tự thích nghi theo context.

### 8.4 Temperature — điều chỉnh độ ngẫu nhiên

Temperature $\tau$ được áp dụng trước khi softmax:

$$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

| τ | Hiệu ứng | Dùng khi nào |
|---|---|---|
| < 1 (ví dụ 0.7) | Phân phối sắc hơn, model "tự tin" hơn | Muốn bản dịch nhất quán, bám nghĩa |
| = 1 | Phân phối gốc, không thay đổi | Baseline |
| > 1 | Phân phối phẳng hơn, model "liều" hơn | Muốn sáng tạo, đa dạng |

Với dịch máy: **τ = 0.7 + top_p = 0.9** là cấu hình thường dùng.

### 8.5 Cấu hình generate() cho từng mục đích

```python
# Cấu hình 1: Nhất quán, bám nghĩa (production)
model.generate(**encoded,
    forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"],
    num_beams=4,
    length_penalty=0.6,
    max_length=128)

# Cấu hình 2: Tự nhiên, hơi đa dạng (chatbot, content)
model.generate(**encoded,
    forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"],
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    max_length=128)

# Cấu hình 3: Kết hợp top-p + top-k (cân bằng)
model.generate(**encoded,
    forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"],
    do_sample=True,
    top_p=0.9,
    top_k=50,
    temperature=0.7,
    max_length=128)
```

**Khi kết hợp top-p + top-k:** lấy `min(nucleus_size, k)` token — giới hạn cả hai chiều, thường cho kết quả cân bằng nhất.

---

## 9. Luồng dữ liệu end-to-end

```
OPUS-100 (HuggingFace)
    │
    ▼ src/data/loader.py
Gộp nhiều cặp ngôn ngữ
    {"pair": "en-vi", "src": "...", "tgt": "..."}
    │
    ▼ src/data/preprocessor.py
Tokenize với language code
    {"input_ids": [...], "attention_mask": [...],
     "labels": [...],    "pair": "en-vi"}
    │
    ▼ src/data/collator.py
Tạo batch tensor
    input_ids:      (B, 128)
    attention_mask: (B, 128)
    labels:         (B, 128)   ← có -100 ở vị trí pad
    pair:           ["en-vi", "de-en", ...]
    │
    ▼ src/training/trainer.py — compute_loss
Forward pass → logits (B, 128, 250000)
Tính loss per token → reshape → mask padding
Tính loss per sample → nhân pair_weights
Balanced loss = weighted mean
    │
    ▼ Backward + optimizer step
Cập nhật 610M parameters
    │
    ▼ Mỗi epoch: src/evaluation/metrics.py
generate() → decode → SacreBLEU
    │
    ▼ outputs/
checkpoints/best_model/
bleu_results.json: {"en-vi": 23.4, "de-en": 19.1, "en-fr": 28.7}
    │
    ▼ src/inference/translator.py
translate("I love you", src_lang="en", tgt_lang="vi")
    → "Tôi yêu bạn"
```

---

## 10. Checklist triển khai

### Phase 1 — Setup

- [ ] Cài môi trường: `transformers`, `datasets`, `evaluate`, `sacrebleu`, `torch`
- [ ] Kiểm tra GPU available và VRAM
- [ ] Load thử tokenizer + model, verify không lỗi
- [ ] Load 100 samples OPUS-100, in ra vài cặp để kiểm tra

### Phase 2 — Data pipeline

- [ ] `loader.py`: load được 3 config, gộp thành DatasetDict
- [ ] In Counter để kiểm tra phân phối pair
- [ ] `preprocessor.py`: tokenize 1 sample thủ công, verify shape
- [ ] Verify labels: không có padding_id nào còn sót (phải toàn -100 hoặc token id)
- [ ] `collator.py`: collate 4 samples, kiểm tra shape tensor

### Phase 3 — Model & Loss

- [ ] Load model thành công
- [ ] Tính pair_weights, in ra để verify giá trị hợp lý
- [ ] Forward pass thủ công với 1 batch, kiểm tra logits shape = (B, T, 250000)
- [ ] Compute balanced loss thủ công, verify là scalar

### Phase 4 — Training

- [ ] Khởi tạo BalancedLossSeq2SeqTrainer
- [ ] Chạy 10 steps với tiny dataset để verify không crash
- [ ] Verify train loss giảm sau 10 steps
- [ ] Chạy evaluation thủ công, verify BLEU > 0

### Phase 5 — Full training

- [ ] Chạy 1 epoch với MAX_SAMPLES_PER_PAIR = 10000 (smoke test)
- [ ] Kiểm tra checkpoint được lưu đúng
- [ ] Chạy full training 2 epoch
- [ ] Log BLEU theo từng pair sau mỗi epoch

### Phase 6 — Inference

- [ ] Load checkpoint best model
- [ ] Dịch thử 10 câu với cả 3 config sampling
- [ ] So sánh output: top-p vs top-k vs beam search
- [ ] Verify forced_bos_token_id đúng ngôn ngữ đích

### Lỗi thường gặp và cách fix

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| BLEU = 0 sau eval | `predict_with_generate=False` | Thêm vào TrainingArguments |
| `KeyError: 'pair'` trong loss | `remove_unused_columns=True` | Đặt thành `False` |
| Model sinh sai ngôn ngữ | Sai `forced_bos_token_id` | Dùng `tokenizer.lang_code_to_id["vi_VN"]` |
| OOM khi train | Batch quá lớn | Giảm `batch_size`, bật `gradient_checkpointing` |
| Loss = NaN | LR quá lớn hoặc labels có lỗi | Dùng LR = 5e-5, verify labels không có NaN |
| Val loss tăng sớm | Overfitting | Giảm epoch, thêm `weight_decay=0.01` |
