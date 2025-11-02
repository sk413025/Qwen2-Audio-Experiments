"""
基於 Transformer 的 Domain Adapter 實現

靈感來源：sequence_insertion_explained.py
核心概念：使用序列級的 Transformer 來學習 LDV → 麥克風 的特徵轉換

設計理念：
1. 不改變特徵維度 (1280維保持不變)
2. 在序列層面進行適應 (處理時序關係)
3. 使用 Self-Attention 捕捉音頻內部的依賴關係
4. 殘差連接保持原始信息

架構：
LDV音頻 → audio_tower → [1, seq_len, 1280] → Domain Adapter (Transformer) → [1, seq_len, 1280] → projector
                                              ↑
                                    保持維度，只調整特徵分佈
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import numpy as np
import librosa
from pathlib import Path
import json
import time
from typing import List, Dict, Optional

# ============================================================================
# 第一部分：基於 Transformer 的 Domain Adapter
# ============================================================================

class TransformerDomainAdapter(nn.Module):
    """
    基於 Transformer 的 Domain Adapter

    設計靈感來自序列插入機制：
    - 序列插入：在序列維度操作，保持特徵維度
    - Domain Adapter：同樣在序列維度學習轉換，保持特徵維度

    為什麼使用 Transformer？
    1. 音頻是時序數據，不同時間步之間有依賴關係
    2. Self-Attention 可以捕捉長距離依賴
    3. 就像序列插入後，attention 讓音頻 embeddings 互相關聯
    4. Transformer 可以學習「哪些時間步需要調整，哪些不需要」

    工作原理：
    ```
    輸入: [batch, seq_len, 1280]  (LDV 特徵)
      ↓
    Position Encoding (添加位置信息)
      ↓
    N × Transformer Encoder Layer
      ├─ Multi-Head Self-Attention  (捕捉時序關係)
      └─ Feed-Forward Network        (特徵轉換)
      ↓
    輸出: [batch, seq_len, 1280]  (麥克風風格特徵)

    + 殘差連接: output = input + scale * adapter_output
    ```
    """

    def __init__(
        self,
        input_dim: int = 1280,          # audio_tower 輸出維度
        num_layers: int = 2,            # Transformer 層數
        num_heads: int = 8,             # 注意力頭數
        ff_dim: int = 2048,             # Feed-forward 隱藏層維度
        dropout: float = 0.1,           # Dropout 比率
        max_seq_len: int = 3000,        # 最大序列長度
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers

        # 位置編碼（重要！音頻是時序數據）
        # 類似於序列插入後，每個位置都有位置信息
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, input_dim)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # [batch, seq, features]
            norm_first=True    # Pre-LayerNorm (更穩定)
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 輸出投影（可選，保持維度不變）
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

        # 可學習的縮放因子（控制 adapter 的影響程度）
        # 初始化為小值，避免一開始就破壞原始特徵
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        # Gate 機制（可選）：讓模型學習「每個時間步是否需要調整」
        self.use_gate = True
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )

    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            audio_features: [batch_size, seq_len, input_dim]
                           來自 audio_tower 的特徵
            attention_mask: [batch_size, seq_len]
                           可選的注意力 mask

        Returns:
            adapted_features: [batch_size, seq_len, input_dim]
                            調整後的特徵

        流程類比序列插入：
        1. 就像文字 tokens 有位置編碼，音頻特徵也需要位置信息
        2. Self-Attention 讓每個時間步關注其他時間步（類似於 attention 讓音頻 embeddings 互相關聯）
        3. 學習特徵調整，但保持維度不變
        """
        batch_size, seq_len, _ = audio_features.shape

        # 1. 添加位置編碼
        # 類似於序列插入後，每個位置都有清晰的位置信息
        pos_enc = self.pos_encoding[:, :seq_len, :]
        features_with_pos = audio_features + pos_enc

        # 2. 準備 attention mask（如果有）
        # PyTorch Transformer 期望 mask 格式：[seq_len, seq_len]
        if attention_mask is not None:
            # 將 [batch, seq_len] 轉換為 [batch, seq_len, seq_len]
            # True 表示被 mask（不關注），False 表示正常關注
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # 3. 通過 Transformer
        # Self-Attention 讓模型學習：
        # - 哪些時間步需要參考其他時間步
        # - 如何調整特徵分佈
        # - LDV 和麥克風特徵的差異模式
        transformed = self.transformer(
            features_with_pos,
            src_key_padding_mask=key_padding_mask
        )

        # 4. 輸出投影
        adapter_output = self.output_proj(transformed)

        # 5. Gate 機制（可選）
        if self.use_gate:
            gate_values = self.gate(audio_features)
            adapter_output = gate_values * adapter_output

        # 6. 殘差連接 + 縮放
        # 保留原始信息，只添加必要的調整
        # output = input + scale * Δ
        adapted_features = audio_features + self.scale * adapter_output

        return adapted_features

    def get_num_params(self):
        """計算參數量"""
        return sum(p.numel() for p in self.parameters())


class LightweightDomainAdapter(nn.Module):
    """
    輕量級 Domain Adapter（更小的版本）

    使用場景：
    - 數據較少時
    - 計算資源有限時
    - LDV 和麥克風差異較小時

    架構簡化：
    - 只有 1 層 Transformer
    - 較少的 attention heads
    - 較小的 FFN
    """

    def __init__(
        self,
        input_dim: int = 1280,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim

        # 簡化的單層 Transformer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(input_dim)

        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        # Self-attention
        attn_out, _ = self.attention(
            audio_features, audio_features, audio_features
        )
        features = self.norm1(audio_features + attn_out)

        # Feed-forward
        ffn_out = self.ffn(features)
        features = self.norm2(features + ffn_out)

        # 殘差連接
        return audio_features + self.scale * features

    def get_num_params(self):
        """計算參數量"""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# 第二部分：將 Domain Adapter 插入 Qwen2-Audio
# ============================================================================

class Qwen2AudioWithDomainAdapter(nn.Module):
    """
    帶有 Domain Adapter 的 Qwen2-Audio

    架構：
    LDV音頻 → audio_tower → [Domain Adapter] → projector → LLM
                           ↑ 插入在這裡

    類比序列插入：
    - 序列插入：在特定位置插入音頻 embeddings
    - Domain Adapter：在特定位置插入特徵轉換模組
    """

    def __init__(
        self,
        base_model: Qwen2AudioForConditionalGeneration,
        adapter_type: str = "transformer",  # "transformer" or "lightweight"
        **adapter_kwargs
    ):
        super().__init__()

        self.base_model = base_model

        # 創建 Domain Adapter
        if adapter_type == "transformer":
            self.domain_adapter = TransformerDomainAdapter(**adapter_kwargs)
        elif adapter_type == "lightweight":
            self.domain_adapter = LightweightDomainAdapter(**adapter_kwargs)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        # 凍結原始模型的所有參數
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 只訓練 domain adapter
        for param in self.domain_adapter.parameters():
            param.requires_grad = True

        print(f"\n✓ Domain Adapter 已創建")
        print(f"  類型: {adapter_type}")
        print(f"  參數量: {self.domain_adapter.get_num_params():,}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        前向傳播，插入 Domain Adapter

        流程：
        1. 如果有音頻特徵，通過 audio_tower
        2. 通過 Domain Adapter 調整特徵
        3. 繼續原始模型的流程
        """

        # 如果有音頻特徵，需要先處理
        if input_features is not None:
            # 1. 通過 audio_tower
            # 注意：這需要訪問 base_model 的內部
            # 實際實現可能需要修改

            # 獲取 audio_tower
            audio_tower = self.base_model.audio_tower

            # 提取音頻特徵
            with torch.no_grad():  # audio_tower 是凍結的
                audio_features = audio_tower(input_features)

            # 2. 通過 Domain Adapter（關鍵步驟！）
            audio_features = self.domain_adapter(
                audio_features,
                attention_mask=feature_attention_mask
            )

            # 3. 將調整後的特徵傳給後續流程
            # 注意：這裡需要替換原始的音頻特徵
            # 實際實現中，可能需要修改 base_model 的 forward
            # 這裡簡化處理

            # 直接調用 base_model，讓它處理後續流程
            # 注意：這是簡化版本，實際需要更複雜的集成
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # 傳入調整後的音頻特徵
                # 實際實現可能需要不同的參數名
                labels=labels,
                **kwargs
            )
        else:
            # 沒有音頻，直接使用原始模型
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

    def generate(self, **kwargs):
        """生成方法"""
        return self.base_model.generate(**kwargs)


# ============================================================================
# 第三部分：更實用的集成方法（Monkey Patching）
# ============================================================================

def add_domain_adapter_to_model(
    model: Qwen2AudioForConditionalGeneration,
    adapter_type: str = "transformer",
    **adapter_kwargs
):
    """
    通過 monkey patching 將 Domain Adapter 添加到模型

    這是一個更實用的方法，不需要重寫整個 forward

    Args:
        model: 原始的 Qwen2-Audio 模型
        adapter_type: adapter 類型
        **adapter_kwargs: adapter 的參數

    Returns:
        修改後的模型
    """
    print("\n" + "="*100)
    print("  添加 Domain Adapter（Monkey Patching 方式）")
    print("="*100)

    # 1. 創建 Domain Adapter
    if adapter_type == "transformer":
        domain_adapter = TransformerDomainAdapter(**adapter_kwargs)
    elif adapter_type == "lightweight":
        domain_adapter = LightweightDomainAdapter(**adapter_kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    # 2. 將 adapter 添加為模型的屬性
    model.domain_adapter = domain_adapter

    # 3. 保存原始的 audio_tower forward
    original_audio_tower_forward = model.audio_tower.forward

    # 4. 定義新的 forward，插入 Domain Adapter
    def audio_tower_with_adapter(self, input_features):
        """新的 audio_tower forward，包含 Domain Adapter"""
        # 調用原始的 audio_tower
        audio_features = original_audio_tower_forward(input_features)

        # 通過 Domain Adapter
        audio_features = model.domain_adapter(audio_features)

        return audio_features

    # 5. 替換 audio_tower 的 forward 方法
    import types
    model.audio_tower.forward = types.MethodType(
        audio_tower_with_adapter,
        model.audio_tower
    )

    # 6. 凍結所有原始參數
    for name, param in model.named_parameters():
        if 'domain_adapter' not in name:
            param.requires_grad = False

    # 7. 只訓練 domain adapter
    for param in model.domain_adapter.parameters():
        param.requires_grad = True

    # 8. 統計參數
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n✓ Domain Adapter 已添加")
    print(f"  類型: {adapter_type}")
    print(f"  位置: audio_tower → [Domain Adapter] → projector")
    print(f"  參數量: {domain_adapter.get_num_params():,} ({domain_adapter.get_num_params()/1e6:.2f}M)")
    print(f"\n參數統計:")
    print(f"  總參數: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  可訓練: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  比例: {trainable_params/total_params*100:.4f}%")

    print("\n架構:")
    print("  LDV音頻 → audio_tower → [Domain Adapter] → projector → LLM")
    print("                          ↑ 新插入的模組")
    print("="*100)

    return model


# ============================================================================
# 第四部分：訓練相關（重用現有代碼）
# ============================================================================

# 可以重用 stage1_lora_training.py 中的：
# - Qwen2AudioTrainingDataset
# - load_dataset_from_disk
# - collate_fn
# - validate
# - train_with_lora (重命名為 train_with_adapter)

# 這裡簡化，只展示關鍵差異

def train_with_domain_adapter(
    model,
    processor,
    train_loader,
    val_loader=None,
    num_epochs=3,
    learning_rate=1e-4,
    device='mps'
):
    """
    使用 Domain Adapter 訓練

    與 LoRA 訓練的區別：
    - LoRA: 在 LLM 層添加 adapters
    - Domain Adapter: 在 audio_tower 和 projector 之間添加 adapter

    相同點：
    - 都只訓練少量參數
    - 都保留原始模型不變
    - 都使用相同的訓練循環
    """
    print("\n" + "="*100)
    print("  開始 Domain Adapter 訓練")
    print("="*100)

    model = model.to(device)
    model.train()

    # 優化器（只優化 domain_adapter）
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    if len(trainable_params) == 0:
        print("❌ 錯誤：沒有可訓練參數！")
        return model

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    print(f"\n訓練配置:")
    print(f"  訓練樣本: {len(train_loader.dataset)}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  學習率: {learning_rate}")
    print(f"  可訓練參數: {sum(p.numel() for p in trainable_params):,}")
    print(f"  設備: {device}\n")

    # 訓練循環（與 LoRA 相同）
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                input_features = batch.get('input_features')
                feature_attention_mask = batch.get('feature_attention_mask')
                if input_features is not None:
                    input_features = input_features.to(device)
                if feature_attention_mask is not None:
                    feature_attention_mask = feature_attention_mask.to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

                if (step + 1) % 5 == 0 or step == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{step+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Avg Loss: {avg_loss:.4f}")

            except Exception as e:
                print(f"❌ 步驟 {step} 出錯: {e}")
                import traceback
                traceback.print_exc()
                continue

        epoch_time = time.time() - start_time
        avg_epoch_loss = total_loss / len(train_loader)

        print(f"\n{'='*100}")
        print(f"Epoch {epoch+1}/{num_epochs} 完成")
        print(f"  訓練 Loss: {avg_epoch_loss:.4f}")
        print(f"  耗時: {epoch_time:.2f} 秒")
        print(f"{'='*100}\n")

        # 保存 checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f'checkpoint_domain_adapter_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'domain_adapter_state_dict': model.domain_adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"✓ Checkpoint 已保存: {checkpoint_path}\n")

    print("🎉 Domain Adapter 訓練完成！\n")
    return model


# ============================================================================
# 第五部分：主程式
# ============================================================================

def main():
    """
    主程式：展示如何使用基於 Transformer 的 Domain Adapter
    """

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║      基於 Transformer 的 Domain Adapter                                ║
║                                                                          ║
║      靈感來源：sequence_insertion_explained.py                          ║
║                                                                          ║
║      核心概念：                                                          ║
║        • 序列插入：在序列維度操作，保持特徵維度                         ║
║        • Domain Adapter：同樣在序列維度學習轉換                        ║
║        • 使用 Transformer 捕捉時序關係                                 ║
║                                                                          ║
║      架構：                                                              ║
║        LDV音頻 → audio_tower → [Domain Adapter] → projector → LLM      ║
║                                ↑ Transformer 處理                       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    model_name = "Qwen/Qwen2-Audio-7B-Instruct"

    # ========================================================================
    # 步驟 1: 加載模型
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 1: 加載 Qwen2-Audio 模型")
    print("="*100)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n使用設備: {device}")

    print(f"\n加載模型: {model_name}")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    print("✓ 模型加載完成")

    # ========================================================================
    # 步驟 2: 添加 Domain Adapter
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 2: 添加 Transformer Domain Adapter")
    print("="*100)

    # 選項 1: 完整版 Transformer Adapter
    model = add_domain_adapter_to_model(
        model,
        adapter_type="transformer",
        input_dim=1280,         # audio_tower 輸出維度
        num_layers=2,           # Transformer 層數
        num_heads=8,            # 注意力頭數
        ff_dim=2048,            # Feed-forward 維度
        dropout=0.1
    )

    # 選項 2: 輕量級 Adapter（數據較少時使用）
    # model = add_domain_adapter_to_model(
    #     model,
    #     adapter_type="lightweight",
    #     input_dim=1280,
    #     num_heads=4,
    #     ff_dim=1024,
    #     dropout=0.1
    # )

    # ========================================================================
    # 步驟 3: 驗證設置
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 3: 驗證 Domain Adapter 設置")
    print("="*100)

    # 檢查可訓練參數
    trainable_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)

    print(f"\n可訓練參數:")
    for name in trainable_names[:10]:  # 顯示前 10 個
        print(f"  - {name}")
    if len(trainable_names) > 10:
        print(f"  ... 還有 {len(trainable_names) - 10} 個")

    # ========================================================================
    # 步驟 4: 準備數據並訓練
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 4: 準備訓練（需要真實的 LDV 數據）")
    print("="*100)

    print("\n⚠️  注意：這個腳本展示了架構設計")
    print("實際訓練需要：")
    print("  1. 真實的 LDV 音頻數據（不是合成數據）")
    print("  2. 對應的標註（轉錄文本或對話）")
    print("  3. 如有麥克風數據對比更好")

    print("\n如何使用：")
    print("  1. 準備 LDV 數據集")
    print("  2. 使用 stage1_lora_training.py 的數據加載代碼")
    print("  3. 調用 train_with_domain_adapter()")

    # 示例：如果有數據
    # train_loader = prepare_ldv_dataloader(...)
    # model = train_with_domain_adapter(
    #     model, processor, train_loader,
    #     num_epochs=3, learning_rate=1e-4, device=device
    # )

    print("\n" + "="*100)
    print("  架構設計完成")
    print("="*100)

    print("\n✅ Domain Adapter 的優勢:")
    print("  1. 使用 Transformer 捕捉音頻時序關係")
    print("  2. 類似序列插入，在序列維度操作")
    print("  3. 保持特徵維度不變 (1280 → 1280)")
    print("  4. 完全不修改原始模型")
    print("  5. 可學習的縮放和 gate 機制")

    print("\n與序列插入的類比:")
    print("  序列插入:    在位置維度插入音頻 embeddings")
    print("  Domain Adapter: 在時序維度調整音頻特徵")
    print("  共同點:     都保持特徵維度，都用 Transformer 處理")

    print("\n下一步:")
    print("  1. 準備 LDV 音頻數據")
    print("  2. 運行訓練")
    print("  3. 評估在 LDV 和麥克風音頻上的性能")
    print("  4. 調整 adapter 參數（層數、頭數等）")

    print("="*100)


if __name__ == "__main__":
    main()
