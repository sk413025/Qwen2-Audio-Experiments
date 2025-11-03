# Spatial Token 對齊策略
## 如何在保留 Audio Projector 的同時添加空間信息

**核心問題**: Spatial Token 表示方向信息，如何對齊到 LLM 空間，且不破壞已有的音頻內容對齊？

---

## 🔍 理解現有的對齊機制

### Multi-Modal Projector 的作用

```python
# Qwen2-Audio 的現有流程
audio_waveform
    ↓
Whisper Encoder [seq_len, 1280]  # 音頻內容語義
    ↓
Multi-Modal Projector [seq_len, 3584]  # 對齊到 LLM 空間
    ↓
Audio Embeddings (已對齊)

# Projector 訓練的目標
使得: Audio Embeddings 的語義 ≈ Text Embeddings 的語義
例如: "玻璃破碎聲" (音頻) ≈ "玻璃破碎" (文字)
```

**關鍵點**:
- Projector 學習的是 **音頻內容語義** 到 **文字語義** 的對齊
- 訓練數據: (音頻, 描述文字) 配對，例如 (玻璃破碎音頻, "玻璃破碎")
- 對齊的是語義空間，不是任意特徵

### 為什麼 Spatial Token 不能走同一個 Projector？

```
Audio Projector 的映射:
  音頻聲學特徵 [1280] → 音頻內容語義 [3584]
  "玻璃破碎的波形" → "玻璃破碎的概念"

Spatial Features 的性質:
  ITD/ILD 特徵 [256] → ???
  "時間差 660μs，強度差 10dB" → ???

問題:
  1. Spatial Features 不是音頻內容特徵（不是 Whisper 輸出）
  2. Audio Projector 從未見過這種特徵
  3. 強行通過會得到無意義的輸出
```

---

## 🎯 核心洞察：不同模態需要不同對齊

### 類比理解

在 Qwen2-Audio 的序列中：

```
[Text Tokens | Audio Embeddings | Text Tokens]
      ↑              ↑                ↑
   文字模態        音頻模態         文字模態
   (Tokenizer)    (Projector)     (Tokenizer)
```

**觀察**:
- Text Tokens 和 Audio Embeddings 來自**不同的處理流程**
- Text: Tokenizer → Embedding Table
- Audio: Whisper → Projector
- 但它們可以共存在同一個序列中！

**為什麼可以共存**？
- 都映射到了 LLM 的統一空間 [3584 維]
- LLM 通過 Self-Attention 學習它們之間的關係
- 不需要所有模態都經過相同的對齊器

### 應用到 Spatial Token

```
[Text Tokens | Audio Embeddings | Spatial Token | Text Tokens]
      ↑              ↑                  ↑            ↑
   文字模態        音頻內容模態       空間方向模態   文字模態
   (Tokenizer)    (Audio Proj)     (Spatial Proj)  (Tokenizer)
                       ↑                  ↑
                 保留預訓練          新增，需訓練
```

**關鍵**:
- Audio Embeddings 仍然走原有 Projector（完全凍結）
- Spatial Token 走**獨立的** Spatial Projector
- 兩者在序列中**並列**，不相互干擾
- LLM 學習關聯："Audio 說內容是什麼"+"Spatial 說方向在哪"

---

## 💡 方案 A: 輕量級 Spatial Projector (推薦 ⭐⭐⭐⭐⭐)

### 架構設計

```
左聲道 Mel-Spec [128, T]
右聲道 Mel-Spec [128, T]
        ↓
╔═══════════════════════════════════╗
║  Spatial Feature Extractor        ║
║  (ITD/ILD Branches)               ║
║  - 輸出: [batch, 256]              ║
╚═══════════════════════════════════╝
        ↓
╔═══════════════════════════════════╗
║  Spatial Projector (NEW)          ║
║  - 將空間特徵對齊到 LLM 空間       ║
║  - 輸入: [256]                    ║
║  - 輸出: [3584]                   ║
╚═══════════════════════════════════╝
        ↓
  Spatial Token [3584]
```

### 具體實現

```python
class SpatialProjector(nn.Module):
    """
    將空間特徵投影到 LLM 空間

    設計理念: 類似 Audio Projector，但針對空間特徵
    """
    def __init__(self, spatial_dim=256, llm_dim=3584):
        super().__init__()

        # 多層投影（模仿 Audio Projector 的結構）
        self.projector = nn.Sequential(
            nn.Linear(spatial_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),

            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.1),

            nn.Linear(2048, llm_dim),
            nn.LayerNorm(llm_dim)
        )

    def forward(self, spatial_features):
        """
        Args:
            spatial_features: [batch, 256] - ITD/ILD 特徵

        Returns:
            spatial_token: [batch, 3584] - 對齊到 LLM 空間
        """
        return self.projector(spatial_features)


class Qwen2AudioWithSpatialToken(nn.Module):
    """
    完整模型: Audio Projector (凍結) + Spatial Projector (訓練)
    """
    def __init__(self, pretrained_model_name):
        super().__init__()

        # 載入預訓練 Qwen2-Audio
        self.qwen2_audio = Qwen2AudioForConditionalGeneration.from_pretrained(
            pretrained_model_name
        )
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        # === 新增模組 ===

        # 1. 空間特徵提取器
        self.spatial_extractor = SpatialFeatureExtractor(
            n_mels=128,
            output_dim=256
        )

        # 2. 空間投影器（關鍵！）
        self.spatial_projector = SpatialProjector(
            spatial_dim=256,
            llm_dim=self.qwen2_audio.config.text_config.hidden_size
        )

        # 添加 <|SPATIAL|> token
        self.spatial_token_id = self._add_spatial_token()

        # 凍結 Qwen2-Audio（包括 Audio Projector）
        for param in self.qwen2_audio.parameters():
            param.requires_grad = False

    def _add_spatial_token(self):
        """添加 <|SPATIAL|> 特殊 token"""
        new_tokens = ['<|SPATIAL|>']
        self.processor.tokenizer.add_special_tokens(
            {'additional_special_tokens': new_tokens}
        )

        # 調整 embedding 層
        self.qwen2_audio.resize_token_embeddings(
            len(self.processor.tokenizer)
        )

        return self.processor.tokenizer.convert_tokens_to_ids('<|SPATIAL|>')

    def forward(self, left_audio, right_audio, text_prompt):
        """
        關鍵: 兩條獨立的路徑
        """
        device = self.qwen2_audio.device

        # ═════════════════════════════════════════
        # 路徑 1: 音頻內容（保留原有流程）
        # ═════════════════════════════════════════

        # 混合音頻（用於內容理解）
        if isinstance(left_audio, torch.Tensor):
            mixed_audio = (left_audio + right_audio) / 2
            mixed_audio = mixed_audio.cpu().numpy()
        else:
            mixed_audio = (left_audio + right_audio) / 2

        # ═════════════════════════════════════════
        # 路徑 2: 空間方向（新增流程）
        # ═════════════════════════════════════════

        with torch.no_grad():
            left_mel = self.extract_mel(left_audio)
            right_mel = self.extract_mel(right_audio)

        # 提取空間特徵
        spatial_features = self.spatial_extractor(left_mel, right_mel)  # [1, 256]

        # 投影到 LLM 空間（關鍵步驟！）
        spatial_token = self.spatial_projector(spatial_features)  # [1, 3584]
        spatial_token = spatial_token.unsqueeze(1)  # [1, 1, 3584] (加入 seq 維度)

        # ═════════════════════════════════════════
        # 構建提示（包含 <|SPATIAL|> token）
        # ═════════════════════════════════════════

        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "mixed.wav"},
                {"type": "text", "text": "<|SPATIAL|>"},  # 佔位符
                {"type": "text", "text": text_prompt}
            ]
        }]

        text_with_spatial = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # 處理輸入
        inputs = self.processor(
            text=text_with_spatial,
            audios=[mixed_audio],
            return_tensors="pt",
            sampling_rate=16000
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ═════════════════════════════════════════
        # Hook: 在序列插入時替換 <|SPATIAL|> token
        # ═════════════════════════════════════════

        def embedding_hook(module, input_ids, output):
            """
            在 embedding 層替換 <|SPATIAL|> token

            output: [batch, seq_len, 3584] - Text Embeddings
            """
            # 找到 <|SPATIAL|> 的位置
            spatial_positions = (input_ids[0] == self.spatial_token_id)

            if spatial_positions.any():
                # 替換為 Spatial Token
                spatial_idx = spatial_positions.nonzero(as_tuple=True)[0]
                for idx in spatial_idx:
                    output[:, idx:idx+1, :] = spatial_token.to(output.device)

            return output

        # 註冊 hook
        handle = self.qwen2_audio.language_model.get_input_embeddings().register_forward_hook(
            lambda module, input, output: embedding_hook(module, inputs['input_ids'], output)
        )

        # ═════════════════════════════════════════
        # 生成
        # ═════════════════════════════════════════

        try:
            output_ids = self.qwen2_audio.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
        finally:
            handle.remove()

        return output_ids
```

### 序列結構

```
最終序列:
[system_tokens | <audio_bos> | Audio_Embs_1...187 | <audio_eos> |
 Spatial_Token | text_tokens | ...]

其中:
  Audio_Embs: 來自 Whisper → Audio Projector (凍結)
              表示音頻內容 "玻璃破碎"

  Spatial_Token: 來自 Spatial Extractor → Spatial Projector (訓練)
                 表示空間方向 "60度"

  兩者在序列中並列，不相互干擾
```

---

## 🎓 訓練策略

### 階段 1: 訓練 Spatial Projector (10-20 epochs)

**目標**: 讓 Spatial Token 能被 LLM 理解為方向信息

```python
# 凍結 Qwen2-Audio (包括 Audio Projector)
for param in model.qwen2_audio.parameters():
    param.requires_grad = False

# 只訓練空間模組
optimizer = AdamW([
    {'params': model.spatial_extractor.parameters(), 'lr': 1e-4},
    {'params': model.spatial_projector.parameters(), 'lr': 1e-4},  # 關鍵
], weight_decay=0.01)

# 訓練數據
# - 使用 HRTF 生成的合成數據
# - Ground truth: 方向描述文字
# - 讓 LLM 學習理解 Spatial Token

# 損失: 標準的 Language Modeling Loss
loss = CrossEntropyLoss(logits, labels)
```

**訓練樣本範例**:

```json
{
    "left_audio": "sample_001_left.wav",
    "right_audio": "sample_001_right.wav",
    "angle": 60.0,
    "conversation": [
        {
            "role": "user",
            "content": [
                {"type": "audio"},  // 混合音頻
                {"type": "text", "text": "<|SPATIAL|>"},  // Spatial Token
                {"type": "text", "text": "請描述這段音頻的內容和方向。"}
            ]
        },
        {
            "role": "assistant",
            "content": "這是玻璃破碎的聲音，來自右側大約 60 度的方向。"
        }
    ]
}
```

**訓練過程**:

```
輸入序列:
[text | Audio_Embs (玻璃破碎，凍結) | Spatial_Token (60度，訓練) | 問題文字]

LLM 生成:
"這是玻璃破碎的聲音，來自右側大約 60 度的方向。"

損失計算:
僅針對生成的文字計算 loss

梯度回傳:
- Audio Embs: 凍結，無梯度
- Spatial Token: 有梯度 ← 更新 Spatial Projector
- LLM 參數: 凍結（階段 1）

結果:
Spatial Projector 學習到將 [256] 的 ITD/ILD 特徵
映射為 [3584] 的 "方向語義" token
```

### 階段 2: LoRA 微調 LLM (10 epochs, 可選)

如果發現 LLM 難以理解 Spatial Token，進行輕量微調：

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# 應用 LoRA
model.qwen2_audio.language_model = get_peft_model(
    model.qwen2_audio.language_model,
    lora_config
)

# 現在 LLM 有 ~16M 可訓練參數
optimizer = AdamW([
    {'params': model.spatial_extractor.parameters(), 'lr': 5e-5},
    {'params': model.spatial_projector.parameters(), 'lr': 5e-5},
    {'params': model.qwen2_audio.language_model.parameters(), 'lr': 2e-5},  # LoRA
])
```

---

## 📊 為什麼這個方案不會破壞 Audio Projector？

### 數學證明

**原始序列** (沒有 Spatial Token):
```
S_original = [Text_1, ..., Text_n, Audio_1, ..., Audio_187, Text_n+1, ...]

其中 Audio_i = AudioProjector(Whisper(mixed_audio))
```

**新序列** (添加 Spatial Token):
```
S_new = [Text_1, ..., Text_n, Audio_1, ..., Audio_187, Spatial, Text_n+1, ...]

其中:
  Audio_i = AudioProjector(Whisper(mixed_audio))  ← 完全不變
  Spatial = SpatialProjector(SpatialExtractor(left, right))  ← 新增
```

**關鍵觀察**:

1. **Audio Embeddings 不變**
   - 仍然通過原有的 Whisper → AudioProjector
   - 所有參數凍結
   - 輸出的 Audio_1...Audio_187 與原始模型完全相同

2. **在序列中的位置**
   - Audio Embeddings 和 Spatial Token 在**不同位置**
   - 不是相加、不是拼接特徵維度
   - 只是在序列軸上並列

3. **LLM 的理解**
   ```
   LLM 看到的序列:
   [..., Audio_187, Spatial, Text_n+1, ...]

   在 Self-Attention 中:
   - Text_n+1 可以 attend 到 Audio_187 (音頻內容)
   - Text_n+1 可以 attend 到 Spatial (方向信息)
   - Audio_187 和 Spatial 可以互相 attend

   但 Audio_187 本身沒有被修改！
   ```

### 類比理解

就像在對話中添加圖片標註：

```
原始對話:
User: [圖片: 一隻貓] 這是什麼動物？
Assistant: 這是一隻貓。

添加位置信息:
User: [圖片: 一隻貓] [位置: 沙發上] 這是什麼動物，在哪裡？
Assistant: 這是一隻貓，它在沙發上。

觀察:
- [圖片] 的內容理解沒有改變（仍然是 "一隻貓"）
- [位置] 是額外的信息（"沙發上"）
- 兩者共存，互不干擾
```

---

## 🔬 Spatial Projector 的對齊機制

### 對齊的本質

```
Audio Projector 的對齊:
  學習: "音頻波形特徵" → "音頻語義概念"
  例子: 玻璃破碎的 Whisper 特徵 → "破碎聲" 的 LLM 表示

Spatial Projector 的對齊:
  學習: "ITD/ILD 特徵" → "方向語義概念"
  例子: (ITD=660μs, ILD=10dB) → "右側 60 度" 的 LLM 表示
```

### 訓練數據驅動對齊

```python
# 訓練樣本
Input:
  - Audio Content: "玻璃破碎" (來自 AudioProjector，凍結)
  - Spatial Token: [s1, s2, ..., s3584] (來自 SpatialProjector，訓練中)
  - Question: "請描述音頻內容和方向"

Target Output:
  "這是玻璃破碎的聲音，來自右側大約 60 度的方向。"

訓練過程:
  1. LLM 從 Audio Embeddings 理解內容 → "玻璃破碎"
  2. LLM 從 Spatial Token 理解方向 → "右側 60 度"
  3. 如果 Spatial Token 無法表達方向，損失會很大
  4. 梯度回傳，更新 SpatialProjector 的參數
  5. 逐漸學習到: 某種 Spatial Token 模式 → "右側 60 度"
```

### 為什麼不需要預訓練對齊數據？

```
Audio Projector 需要預訓練:
  - 因為要學習複雜的 "音頻語義" 對齊
  - 需要大量 (音頻, 描述) 配對數據
  - 例如: (狗叫音頻, "狗叫聲"), (音樂, "鋼琴演奏"), ...

Spatial Projector 不需要預訓練:
  - 方向信息相對簡單: 就是角度
  - 訓練數據容易生成: HRTF 模擬
  - 對齊目標明確: ITD/ILD → 角度
  - 可以端到端訓練（與 LLM 微調一起）
```

---

## 🎯 方案總結

### 關鍵設計決策

| 模組 | 是否訓練 | 作用 | 保留程度 |
|------|---------|------|---------|
| Whisper Encoder | ❌ 凍結 | 音頻特徵提取 | 100% 保留 |
| **Audio Projector** | ❌ 凍結 | 音頻內容對齊 | **100% 保留** ⭐ |
| Spatial Extractor | ✅ 訓練 | ITD/ILD 提取 | 新增 |
| **Spatial Projector** | ✅ 訓練 | 方向信息對齊 | 新增 ⭐⭐ |
| Qwen2 LLM | ⚠️ LoRA (可選) | 語言理解生成 | 99% 保留 |

### 可訓練參數

```
Spatial Extractor:  ~2M
Spatial Projector:  ~8M  (256→1024→2048→3584)
LoRA (可選):        ~16M (r=16)
──────────────────────────
總計:               ~10M (不含 LoRA)
                   ~26M (含 LoRA)

相比 Qwen2-Audio 7B: 0.14% - 0.37%
```

### 序列示意圖

```
最終的 LLM 輸入序列:

位置:  0   1   ...  50   51  ...  237  238  239  240  241  ...
Token: <s> <im> ... <ab> A_1 ... A_187 <ae> S   這   是   ...
                         ↑            ↑    ↑
                    Audio Embs    Spatial  Text
                    (凍結路徑)    (訓練路徑)

關鍵:
  A_1...A_187: 來自 AudioProjector，表示 "玻璃破碎" 的內容語義
  S: 來自 SpatialProjector，表示 "60度右側" 的方向語義

  兩者在序列中並列，LLM 通過 Attention 關聯它們
```

---

## 📝 實現步驟

1. **實現 Spatial Feature Extractor** (ITD/ILD branches)
2. **實現 Spatial Projector** (256→3584 的對齊)
3. **添加 <|SPATIAL|> token 到 tokenizer**
4. **實現 embedding hook** (替換 <|SPATIAL|> 為 Spatial Token)
5. **生成訓練數據** (HRTF + 方向描述文字)
6. **階段 1 訓練**: 凍結 Qwen2-Audio，訓練 Spatial 模組
7. **階段 2 訓練** (可選): LoRA 微調 LLM

---

## ✅ 總結

### 回答你的核心問題

> "關鍵在於怎麼對齊 Spatial: 獨立 Token？"

**答案**: 訓練一個獨立的 **Spatial Projector**，將 ITD/ILD 特徵 [256] 映射到 LLM 空間 [3584]

> "現有的 multimodal projector 要怎麼保留原先已經對齊過的資訊？"

**答案**:
1. **完全凍結 Audio Projector**，不修改任何參數
2. Spatial Token 在序列中是**獨立位置**，不與 Audio Embeddings 混合
3. 就像 Text Token 和 Audio Embeddings 可以共存一樣，Spatial Token 也可以共存
4. LLM 通過 Self-Attention 學習三者的關聯，不需要它們經過同一個 Projector

### 為什麼這個方案有效？

1. ✅ **Audio Projector 完全不動** - 預訓練對齊 100% 保留
2. ✅ **Spatial Projector 獨立訓練** - 學習方向語義對齊
3. ✅ **序列並列，不混合** - 各模態獨立，LLM 學習關聯
4. ✅ **訓練數據可生成** - HRTF 模擬，不需要大量標註

---

**下一步**: 開始實現 Spatial Projector！
