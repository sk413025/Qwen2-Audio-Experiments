"""
Qwen2-Audio 空間音頻定位模型

包含:
- Spatial Feature Extractor (ITD/ILD)
- Spatial Projector (對齊到 LLM 空間)
- LoRA 微調支援
"""

import torch
import torch.nn as nn
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

from spatial_feature_extractor import SpatialFeatureExtractor


class SpatialProjector(nn.Module):
    """
    空間投影器

    將空間特徵 [256] 投影到 LLM 空間 [3584]
    類似 Audio Projector 的作用
    """
    def __init__(self, spatial_dim=256, llm_dim=3584):
        super().__init__()

        # 多層投影（模仿 Audio Projector 結構）
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
            spatial_token: [batch, 3584] - 對齊到 LLM 空間的 token
        """
        return self.projector(spatial_features)


class Qwen2AudioWithSpatialToken(nn.Module):
    """
    Qwen2-Audio 擴展：支援空間音頻定位

    架構:
    1. Audio Content: 混合音頻 → Whisper → Audio Projector (凍結)
    2. Spatial Direction: 雙聲道 → Spatial Extractor → Spatial Projector (訓練)
    3. 序列: [text | audio_embs | <|SPATIAL|> | text]
    """
    def __init__(
        self,
        pretrained_model_name="Qwen/Qwen2-Audio-7B-Instruct",
        spatial_dim=256,
        use_lora=False,
        lora_r=16,
        lora_alpha=32
    ):
        super().__init__()

        print(f"\n{'='*80}")
        print(f"  初始化 Qwen2AudioWithSpatialToken")
        print(f"{'='*80}")

        # 載入預訓練 Qwen2-Audio
        print(f"\n載入預訓練模型: {pretrained_model_name}")
        self.qwen2_audio = Qwen2AudioForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float32,
            device_map=None  # 手動管理設備
        )

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        # 獲取 LLM 維度
        self.llm_dim = self.qwen2_audio.config.text_config.hidden_size
        print(f"  LLM 維度: {self.llm_dim}")

        # === 新增模組 ===

        print(f"\n創建空間感知模組:")

        # 1. 空間特徵提取器
        self.spatial_extractor = SpatialFeatureExtractor(
            n_mels=128,
            output_dim=spatial_dim
        )
        print(f"  ✓ Spatial Feature Extractor (輸出: {spatial_dim})")

        # 2. 空間投影器
        self.spatial_projector = SpatialProjector(
            spatial_dim=spatial_dim,
            llm_dim=self.llm_dim
        )
        print(f"  ✓ Spatial Projector ({spatial_dim} → {self.llm_dim})")

        # 添加 <|SPATIAL|> token
        self.spatial_token_id = self._add_spatial_token()
        print(f"  ✓ <|SPATIAL|> token (ID: {self.spatial_token_id})")

        # 凍結 Qwen2-Audio
        self._freeze_pretrained()

        # 應用 LoRA（可選）
        if use_lora:
            self._apply_lora(lora_r, lora_alpha)

        print(f"\n{'='*80}")

    def _add_spatial_token(self):
        """添加 <|SPATIAL|> 特殊 token"""
        # 檢查是否已存在
        if '<|SPATIAL|>' in self.processor.tokenizer.get_vocab():
            spatial_token_id = self.processor.tokenizer.convert_tokens_to_ids('<|SPATIAL|>')
            return spatial_token_id

        # 添加新 token
        new_tokens = ['<|SPATIAL|>']
        num_added = self.processor.tokenizer.add_special_tokens(
            {'additional_special_tokens': new_tokens}
        )

        if num_added > 0:
            # 調整 embedding 層大小
            self.qwen2_audio.resize_token_embeddings(
                len(self.processor.tokenizer)
            )

        spatial_token_id = self.processor.tokenizer.convert_tokens_to_ids('<|SPATIAL|>')
        return spatial_token_id

    def _freeze_pretrained(self):
        """凍結預訓練的 Qwen2-Audio（包括 Audio Projector）"""
        for param in self.qwen2_audio.parameters():
            param.requires_grad = False

        print(f"\n  ✓ Qwen2-Audio 已凍結（包括 Audio Projector）")

    def _apply_lora(self, r=16, alpha=32):
        """應用 LoRA 微調"""
        print(f"\n應用 LoRA 微調:")
        print(f"  - Rank (r): {r}")
        print(f"  - Alpha: {alpha}")

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.qwen2_audio.language_model = get_peft_model(
            self.qwen2_audio.language_model,
            lora_config
        )

        # 統計 LoRA 參數
        lora_params = sum(
            p.numel() for p in self.qwen2_audio.language_model.parameters()
            if p.requires_grad
        )

        print(f"  ✓ LoRA 參數: {lora_params:,}")

    def get_trainable_params(self):
        """統計可訓練參數"""
        spatial_extractor_params = sum(
            p.numel() for p in self.spatial_extractor.parameters()
            if p.requires_grad
        )

        spatial_projector_params = sum(
            p.numel() for p in self.spatial_projector.parameters()
            if p.requires_grad
        )

        lora_params = sum(
            p.numel() for p in self.qwen2_audio.parameters()
            if p.requires_grad
        )

        total_trainable = spatial_extractor_params + spatial_projector_params + lora_params

        return {
            'spatial_extractor': spatial_extractor_params,
            'spatial_projector': spatial_projector_params,
            'lora': lora_params,
            'total': total_trainable
        }

    def extract_mel(self, audio, device='cpu'):
        """
        提取 Mel-Spectrogram

        Args:
            audio: numpy array [samples] 或 list of arrays
            device: 設備

        Returns:
            mel: [batch, n_mels, T]
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if isinstance(audio, np.ndarray):
            audio = [audio]

        features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        return features.input_features.to(device)

    def forward(self, left_audio, right_audio, text_prompt, text_response=None, labels=None):
        """
        前向傳播

        Args:
            left_audio: [batch, samples] 或 list of numpy arrays
            right_audio: [batch, samples] 或 list of numpy arrays
            text_prompt: str 或 list of str
            text_response: str 或 list of str (訓練時需要，用於構建完整對話)
            labels: [batch, seq_len] 用於訓練的標籤（可選）

        Returns:
            output: ModelOutput 包含 loss, logits 等
        """
        device = next(self.parameters()).device

        # ═════════════════════════════════════════
        # 路徑 1: 音頻內容（保留原有流程）
        # ═════════════════════════════════════════

        # 混合音頻
        if isinstance(left_audio, torch.Tensor):
            mixed_audio = (left_audio + right_audio) / 2
            mixed_audio = mixed_audio.cpu().numpy()
        elif isinstance(left_audio, np.ndarray):
            mixed_audio = (left_audio + right_audio) / 2
        else:  # list
            mixed_audio = [(l + r) / 2 for l, r in zip(left_audio, right_audio)]

        # ═════════════════════════════════════════
        # 路徑 2: 空間方向（新增流程）
        # ═════════════════════════════════════════

        # 提取 Mel-Spectrogram
        with torch.no_grad():
            left_mel = self.extract_mel(left_audio, device)
            right_mel = self.extract_mel(right_audio, device)

        # 提取空間特徵
        spatial_features = self.spatial_extractor(left_mel, right_mel)  # [batch, 256]

        # 投影到 LLM 空間
        spatial_token = self.spatial_projector(spatial_features)  # [batch, 3584]
        spatial_token = spatial_token.unsqueeze(1)  # [batch, 1, 3584]

        # ═════════════════════════════════════════
        # 構建提示（包含 <|SPATIAL|> token）
        # ═════════════════════════════════════════

        # 如果是單個字串，轉為列表
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]

        if text_response is not None and isinstance(text_response, str):
            text_response = [text_response]

        # 構建對話
        conversations = []
        if text_response is not None:
            # 訓練模式：包含 assistant 回答
            for prompt, response in zip(text_prompt, text_response):
                conversations.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": "audio.wav"},
                            {"type": "text", "text": "<|SPATIAL|>"},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ])
        else:
            # 推理模式：只有 user prompt
            for prompt in text_prompt:
                conversations.append([{
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "audio.wav"},
                        {"type": "text", "text": "<|SPATIAL|>"},
                        {"type": "text", "text": prompt}
                    ]
                }])

        # 處理文字
        texts = []
        for conv in conversations:
            text = self.processor.apply_chat_template(
                conv,
                add_generation_prompt=(text_response is None),  # 只在推理時添加 generation prompt
                tokenize=False
            )
            texts.append(text)

        # 準備輸入
        inputs = self.processor(
            text=texts,
            audios=[mixed_audio] if not isinstance(mixed_audio, list) else mixed_audio,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ═════════════════════════════════════════
        # Hook: 替換 <|SPATIAL|> token
        # ═════════════════════════════════════════

        spatial_token_inserted = False

        def embedding_hook(module, args, output):
            """在 embedding 層替換 <|SPATIAL|> token"""
            nonlocal spatial_token_inserted

            if spatial_token_inserted:
                return output

            # 找到 <|SPATIAL|> 的位置
            input_ids = inputs['input_ids']

            for batch_idx in range(input_ids.size(0)):
                spatial_positions = (input_ids[batch_idx] == self.spatial_token_id)

                if spatial_positions.any():
                    spatial_idx = spatial_positions.nonzero(as_tuple=True)[0]
                    for idx in spatial_idx:
                        output[batch_idx, idx:idx+1, :] = spatial_token[batch_idx:batch_idx+1]

            spatial_token_inserted = True
            return output

        # 註冊 hook
        embedding_layer = self.qwen2_audio.language_model.get_input_embeddings()
        handle = embedding_layer.register_forward_hook(embedding_hook)

        # ═════════════════════════════════════════
        # 前向傳播
        # ═════════════════════════════════════════

        try:
            if text_response is not None:
                # 訓練模式：創建 labels
                labels = inputs['input_ids'].clone()
                # 這裡可以進一步 mask 掉 user prompt 部分，只計算 assistant 回答的 loss
                # 目前簡化處理，計算整個序列的 loss
                inputs['labels'] = labels

            output = self.qwen2_audio(**inputs)

        finally:
            handle.remove()
            spatial_token_inserted = False

        return output

    @torch.no_grad()
    def generate(self, left_audio, right_audio, text_prompt, max_new_tokens=256, **kwargs):
        """
        生成回應

        Args:
            left_audio: numpy array [samples]
            right_audio: numpy array [samples]
            text_prompt: str
            max_new_tokens: 最大生成 token 數
            **kwargs: 其他生成參數

        Returns:
            response: str 生成的文字
        """
        self.eval()
        device = next(self.parameters()).device

        # ═════════════════════════════════════════
        # 準備輸入
        # ═════════════════════════════════════════

        # 混合音頻
        if isinstance(left_audio, np.ndarray):
            mixed_audio = (left_audio + right_audio) / 2
        else:
            mixed_audio = [(l + r) / 2 for l, r in zip(left_audio, right_audio)]

        # 提取空間特徵
        with torch.no_grad():
            left_mel = self.extract_mel(left_audio, device)
            right_mel = self.extract_mel(right_audio, device)

        spatial_features = self.spatial_extractor(left_mel, right_mel)
        spatial_token = self.spatial_projector(spatial_features).unsqueeze(1)

        # 構建對話
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "audio.wav"},
                {"type": "text", "text": "<|SPATIAL|>"},
                {"type": "text", "text": text_prompt}
            ]
        }]

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            text=text,
            audios=[mixed_audio] if isinstance(mixed_audio, np.ndarray) else mixed_audio,
            return_tensors="pt",
            sampling_rate=16000
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ═════════════════════════════════════════
        # Hook: 替換 <|SPATIAL|> token
        # ═════════════════════════════════════════

        def embedding_hook(module, args, output):
            input_ids = inputs['input_ids']
            spatial_positions = (input_ids[0] == self.spatial_token_id)

            if spatial_positions.any():
                spatial_idx = spatial_positions.nonzero(as_tuple=True)[0]
                for idx in spatial_idx:
                    output[0, idx:idx+1, :] = spatial_token[0]

            return output

        embedding_layer = self.qwen2_audio.language_model.get_input_embeddings()
        handle = embedding_layer.register_forward_hook(embedding_hook)

        # ═════════════════════════════════════════
        # 生成
        # ═════════════════════════════════════════

        try:
            output_ids = self.qwen2_audio.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get('do_sample', False),
                temperature=kwargs.get('temperature', 1.0),
                top_p=kwargs.get('top_p', 1.0),
            )

            # 只取新生成的部分
            generated_ids = output_ids[:, inputs['input_ids'].size(1):]

            # 解碼
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        finally:
            handle.remove()

        return response


def test_model():
    """測試模型"""
    print("=" * 80)
    print("  測試 Qwen2AudioWithSpatialToken")
    print("=" * 80)

    # 注意：需要預訓練模型
    # 這裡只是測試架構，實際使用需要下載模型
    print("\n注意：此測試需要預訓練的 Qwen2-Audio 模型")
    print("如果模型未下載，請運行：")
    print("  huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct")

    # 統計參數（不實際載入模型）
    print("\n模型架構：")
    print("  1. Spatial Feature Extractor: ~0.9M 參數")
    print("  2. Spatial Projector: ~8.4M 參數")
    print("  3. LoRA (可選): ~16M 參數")
    print("  總計: ~9.3M (不含 LoRA) 或 ~25M (含 LoRA)")


if __name__ == '__main__':
    test_model()
