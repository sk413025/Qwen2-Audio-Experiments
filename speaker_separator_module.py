"""
在 Qwen2-Audio 架構內實現多人語音分離

核心創新：
1. 在 Whisper 特徵層進行分離（不是音頻信號層）
2. 善用 Whisper 的預訓練表示能力
3. 使用 Transformer + Permutation Invariant Training
4. 完全整合在 Qwen2-Audio 架構中

架構：
混合音頻 → audio_tower (Whisper) → [Speaker Separator] → N × projector → N × LLM

設計理念：
- Whisper 已經提取了包含所有說話者信息的特徵
- 在特徵層分離比在音頻信號層更高效
- 重複使用預訓練的 projector 和 LLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from transformers import Qwen2AudioForConditionalGeneration
import numpy as np

# ============================================================================
# 第一部分：Speaker Separator Module（在特徵層分離）
# ============================================================================

class SpeakerSeparatorTransformer(nn.Module):
    """
    基於 Transformer 的說話者分離模組（在 Whisper 特徵層）

    核心概念：
    - 輸入：混合的 Whisper 特徵 [batch, seq_len, 1280]
    - 輸出：N 個分離的特徵 [N, batch, seq_len, 1280]

    設計靈感：
    1. Permutation Invariant Training (PIT) - 說話者順序不重要
    2. Multi-head attention - 捕捉不同說話者的特徵模式
    3. Speaker queries - 類似 DETR，用 query 來「詢問」不同說話者

    善用 Whisper 優勢：
    - Whisper 特徵已包含語義和說話者信息
    - 在高層特徵分離比在原始音頻更容易
    - 可以利用語義線索（不同人說不同內容）
    """

    def __init__(
        self,
        feature_dim: int = 1280,       # Whisper 特徵維度
        max_speakers: int = 4,         # 最大說話者數量
        num_layers: int = 4,           # Transformer 層數
        num_heads: int = 8,            # Attention 頭數
        ff_dim: int = 2048,            # Feed-forward 維度
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.max_speakers = max_speakers
        self.num_layers = num_layers

        # ====================================================================
        # Speaker Queries（關鍵創新！）
        # ====================================================================
        # 類似 DETR 的 object queries
        # 每個 query 對應一個潛在的說話者
        # 模型學習：「第 i 個 query 應該提取第 i 個說話者的特徵」

        self.speaker_queries = nn.Parameter(
            torch.randn(max_speakers, feature_dim)
        )
        nn.init.xavier_uniform_(self.speaker_queries)

        # ====================================================================
        # Cross-Attention Layers
        # ====================================================================
        # 每個 speaker query 通過 cross-attention 從混合特徵中提取信息

        self.cross_attention_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        # ====================================================================
        # Speaker Feature Refinement
        # ====================================================================
        # 對每個說話者的特徵進行精煉

        self.refinement = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads // 2,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(2)
        ])

        # ====================================================================
        # Speaker Activity Detector
        # ====================================================================
        # 檢測每個 speaker query 是否對應真實說話者
        # 輸出：每個 speaker 的活躍概率 [max_speakers]

        self.activity_detector = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
            nn.Sigmoid()
        )

        # ====================================================================
        # Positional Encoding（重用 Whisper 的位置信息）
        # ====================================================================
        # 注意：輸入特徵已經包含 Whisper 的位置編碼
        # 這裡不需要額外添加

        print(f"\n✓ Speaker Separator 已創建")
        print(f"  最大說話者數: {max_speakers}")
        print(f"  Transformer 層數: {num_layers}")
        print(f"  參數量: {self.get_num_params():,}")

    def forward(
        self,
        mixed_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_speakers: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播：從混合特徵中分離出不同說話者

        Args:
            mixed_features: [batch, seq_len, feature_dim]
                           來自 Whisper 的混合特徵
            attention_mask: [batch, seq_len]
                           可選的 mask
            num_speakers: 預期的說話者數量（如果已知）

        Returns:
            separated_features: [batch, max_speakers, seq_len, feature_dim]
                              分離的特徵（每個說話者）
            speaker_probs: [batch, max_speakers]
                          每個說話者的活躍概率

        工作流程：
        1. Speaker queries 通過 cross-attention 從混合特徵提取信息
        2. 每個 query 學習關注特定說話者的特徵
        3. Refinement 進一步分離和增強
        4. Activity detector 判斷哪些 query 對應真實說話者
        """
        batch_size, seq_len, _ = mixed_features.shape

        # ====================================================================
        # 1. 準備 speaker queries
        # ====================================================================
        # 擴展到 batch
        # [max_speakers, feature_dim] → [batch, max_speakers, feature_dim]

        queries = self.speaker_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # ====================================================================
        # 2. Cross-Attention：從混合特徵中提取說話者特徵
        # ====================================================================
        # 每個 query 學習「詢問」：「我對應的說話者在哪裡？」

        # 準備 attention mask
        if attention_mask is not None:
            # 轉換為 transformer 期望的格式
            memory_key_padding_mask = (attention_mask == 0)
        else:
            memory_key_padding_mask = None

        # 初始化每個說話者的特徵為 query
        speaker_features = queries  # [batch, max_speakers, feature_dim]

        # 通過多層 cross-attention
        for layer in self.cross_attention_layers:
            # 對每個說話者 query 進行處理
            # query: [batch, max_speakers, feature_dim]
            # memory: [batch, seq_len, feature_dim] (混合特徵)

            # 重複 mixed_features 以匹配 speaker 數量
            # 這樣每個 speaker query 可以獨立地從混合特徵中提取信息

            speaker_features_new = []
            for i in range(self.max_speakers):
                # 取出第 i 個 speaker 的 query
                query_i = speaker_features[:, i:i+1, :]  # [batch, 1, feature_dim]

                # Cross-attention: query attends to mixed features
                output_i = layer(
                    query_i,                      # tgt
                    mixed_features,               # memory
                    memory_key_padding_mask=memory_key_padding_mask
                )

                speaker_features_new.append(output_i)

            # 合併所有 speaker 的輸出
            speaker_features = torch.cat(speaker_features_new, dim=1)
            # [batch, max_speakers, feature_dim]

        # ====================================================================
        # 3. 擴展到序列維度
        # ====================================================================
        # 現在我們有每個說話者的「查詢結果」
        # 需要擴展到整個序列

        # 方法：使用 speaker_features 作為「風格」特徵
        # 通過 attention 將其應用到混合特徵的每個時間步

        separated_features = []

        for i in range(self.max_speakers):
            # 取出第 i 個 speaker 的特徵向量
            speaker_vec = speaker_features[:, i, :]  # [batch, feature_dim]

            # 計算與混合特徵每個時間步的相似度
            # [batch, feature_dim] × [batch, seq_len, feature_dim]^T
            # = [batch, seq_len]

            similarity = torch.einsum(
                'bd,bsd->bs',
                speaker_vec,
                mixed_features
            ) / np.sqrt(self.feature_dim)

            # Softmax 得到 attention weights
            attn_weights = F.softmax(similarity, dim=-1)  # [batch, seq_len]

            # 使用 attention weights 加權混合特徵
            # 強調屬於該說話者的部分
            speaker_specific = torch.einsum(
                'bs,bsd->bsd',
                attn_weights,
                mixed_features
            )  # [batch, seq_len, feature_dim]

            separated_features.append(speaker_specific)

        # Stack: [max_speakers, batch, seq_len, feature_dim]
        separated_features = torch.stack(separated_features, dim=0)

        # 轉置為 [batch, max_speakers, seq_len, feature_dim]
        separated_features = separated_features.permute(1, 0, 2, 3)

        # ====================================================================
        # 4. Refinement：進一步分離和增強
        # ====================================================================
        # 對每個說話者的特徵序列應用 self-attention

        refined_features = []
        for i in range(self.max_speakers):
            feat = separated_features[:, i, :, :]  # [batch, seq_len, feature_dim]

            # 通過 refinement layers
            for refine_layer in self.refinement:
                feat = refine_layer(feat)

            refined_features.append(feat)

        # Stack: [batch, max_speakers, seq_len, feature_dim]
        separated_features = torch.stack(refined_features, dim=1)

        # ====================================================================
        # 5. Speaker Activity Detection
        # ====================================================================
        # 判斷每個 speaker 是否真實存在
        # 使用 global average pooling + MLP

        speaker_global = torch.mean(separated_features, dim=2)
        # [batch, max_speakers, feature_dim]

        speaker_probs = []
        for i in range(self.max_speakers):
            prob = self.activity_detector(speaker_global[:, i, :])
            # [batch, 1]
            speaker_probs.append(prob)

        speaker_probs = torch.cat(speaker_probs, dim=1)
        # [batch, max_speakers]

        return separated_features, speaker_probs

    def get_num_params(self):
        """計算參數量"""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# 第二部分：整合到 Qwen2-Audio
# ============================================================================

def add_speaker_separator_to_qwen2audio(
    model: Qwen2AudioForConditionalGeneration,
    max_speakers: int = 4,
    **separator_kwargs
):
    """
    將 Speaker Separator 添加到 Qwen2-Audio

    架構變化：
    原始：audio_tower → projector → LLM
    新增：audio_tower → [Speaker Separator] → N × projector → N × LLM

    Args:
        model: Qwen2-Audio 模型
        max_speakers: 最大說話者數量
        **separator_kwargs: Separator 的參數

    Returns:
        修改後的模型
    """
    print("\n" + "="*100)
    print("  添加 Speaker Separator Module 到 Qwen2-Audio")
    print("="*100)

    # ========================================================================
    # 1. 創建 Speaker Separator
    # ========================================================================

    separator = SpeakerSeparatorTransformer(
        feature_dim=1280,  # Whisper 輸出維度
        max_speakers=max_speakers,
        **separator_kwargs
    )

    # 添加為模型屬性
    model.speaker_separator = separator

    # ========================================================================
    # 2. Monkey Patching：修改 audio_tower 的輸出
    # ========================================================================

    # 保存原始 forward
    original_audio_tower_forward = model.audio_tower.forward

    def audio_tower_with_separator(self, input_features):
        """
        新的 audio_tower forward，包含 Speaker Separator

        原始流程：audio → Whisper → features
        新流程：audio → Whisper → [Separator] → N × features
        """
        # 1. 調用原始 Whisper encoder
        mixed_features = original_audio_tower_forward(input_features)
        # [batch, seq_len, 1280]

        # 2. 通過 Speaker Separator
        separated_features, speaker_probs = model.speaker_separator(
            mixed_features
        )
        # separated_features: [batch, max_speakers, seq_len, 1280]
        # speaker_probs: [batch, max_speakers]

        # 3. 篩選活躍的說話者
        # 只保留概率 > threshold 的說話者
        threshold = 0.5
        active_speakers = []

        for i in range(separated_features.shape[1]):
            if torch.mean(speaker_probs[:, i]) > threshold:
                active_speakers.append(separated_features[:, i, :, :])

        # 4. 返回結果
        # 注意：這裡需要特殊處理，因為 Qwen2-Audio 期望單個特徵
        # 我們返回一個特殊的結構

        return {
            'separated_features': separated_features,
            'speaker_probs': speaker_probs,
            'num_active_speakers': len(active_speakers),
            'mixed_features': mixed_features  # 保留原始混合特徵
        }

    # 替換 forward
    import types
    model.audio_tower.forward = types.MethodType(
        audio_tower_with_separator,
        model.audio_tower
    )

    # ========================================================================
    # 3. 凍結原始參數，只訓練 separator
    # ========================================================================

    for name, param in model.named_parameters():
        if 'speaker_separator' not in name:
            param.requires_grad = False

    for param in model.speaker_separator.parameters():
        param.requires_grad = True

    # ========================================================================
    # 4. 統計參數
    # ========================================================================

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n✓ Speaker Separator 已添加")
    print(f"  最大說話者數: {max_speakers}")
    print(f"  位置: audio_tower → [Speaker Separator] → projector")
    print(f"  參數量: {separator.get_num_params():,} ({separator.get_num_params()/1e6:.2f}M)")
    print(f"\n參數統計:")
    print(f"  總參數: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  可訓練: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  比例: {trainable_params/total_params*100:.4f}%")

    print("\n架構:")
    print("  混合音頻 → audio_tower (Whisper) → [Speaker Separator] → ")
    print("  → N × projector → N × LLM → N × 轉錄文本")
    print("                    ↑ 在特徵層分離！")
    print("="*100)

    return model


# ============================================================================
# 第三部分：處理分離後的結果
# ============================================================================

def process_separated_audio(
    model: Qwen2AudioForConditionalGeneration,
    processor,
    audio_path: str,
    instruction: str = "請轉錄這段語音",
    device: str = 'mps'
):
    """
    處理多人語音：分離並轉錄

    Args:
        model: 帶有 Speaker Separator 的 Qwen2-Audio
        processor: Qwen2-Audio processor
        audio_path: 音頻檔案路徑
        instruction: 給 LLM 的指令
        device: 設備

    Returns:
        results: 每個說話者的轉錄結果
    """
    import librosa

    print(f"\n處理音頻: {audio_path}")

    # ========================================================================
    # 1. 加載音頻
    # ========================================================================

    audio, sr = librosa.load(audio_path, sr=16000)

    # ========================================================================
    # 2. 通過 audio_tower + separator（自動分離）
    # ========================================================================

    # 準備輸入
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": instruction}
        ]}
    ]

    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        audios=[audio],
        return_tensors="pt",
        sampling_rate=sr
    )
    inputs = inputs.to(device)

    # 通過 audio_tower（會自動調用 separator）
    with torch.no_grad():
        # 這會返回分離的結果
        separation_result = model.audio_tower(inputs['input_features'])

    separated_features = separation_result['separated_features']
    speaker_probs = separation_result['speaker_probs']
    num_speakers = separation_result['num_active_speakers']

    print(f"\n檢測到 {num_speakers} 個說話者")

    # ========================================================================
    # 3. 對每個說話者進行轉錄
    # ========================================================================

    results = []
    threshold = 0.5

    for i in range(separated_features.shape[1]):
        prob = speaker_probs[0, i].item()

        if prob > threshold:
            print(f"\n處理說話者 {i+1} (活躍概率: {prob:.2f})...")

            # 取出該說話者的特徵
            speaker_feat = separated_features[0, i, :, :]
            # [seq_len, 1280]

            # TODO: 需要修改模型的 forward 來接受預計算的特徵
            # 這裡簡化處理

            results.append({
                'speaker_id': i+1,
                'probability': prob,
                'features': speaker_feat,
                # 'transcription': transcription  # 需要實現
            })

    return results


# ============================================================================
# 第四部分：訓練相關（含合成數據生成）
# ============================================================================

from torch.utils.data import Dataset, DataLoader
from typing import List
import time
import os
import json

class SyntheticMultiSpeakerDataset(Dataset):
    """
    合成多人語音特徵數據集（用於 smoke test）

    支援兩種模式：
    1. 生成新數據並保存到磁碟
    2. 從磁碟加載已保存的數據
    """

    def __init__(
        self,
        num_samples=30,
        max_speakers=3,
        seq_len=100,
        feature_dim=1280,
        seed=42,
        save_dir='training_data',
        dataset_name='synthetic_dataset',
        load_from_disk=False
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_speakers = max_speakers
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.save_dir = save_dir
        self.dataset_name = dataset_name

        # 數據文件路徑
        self.data_path = os.path.join(save_dir, f"{dataset_name}.pt")

        if load_from_disk and os.path.exists(self.data_path):
            # 從磁碟加載
            print(f"從磁碟加載數據: {self.data_path}")
            self.samples = self._load_from_disk()
            print(f"✓ 已加載 {len(self.samples)} 個樣本")
        else:
            # 生成新數據
            np.random.seed(seed)
            torch.manual_seed(seed)

            self.samples = []
            print(f"生成 {num_samples} 個合成樣本...")

            for i in range(num_samples):
                sample = self._generate_sample()
                self.samples.append(sample)
                if (i + 1) % 10 == 0:
                    print(f"  已生成 {i+1}/{num_samples}")

            print("✓ 數據生成完成")

            # 保存到磁碟
            self._save_to_disk()
            print(f"✓ 數據已保存到: {self.data_path}")

    def _generate_sample(self):
        """生成一個合成樣本（模擬 Whisper 特徵）"""
        num_speakers = np.random.randint(2, self.max_speakers + 1)
        separated_features = []

        for i in range(num_speakers):
            # 為每個說話者創建獨特的正弦波模式
            base_freq = (i + 1) * 0.1
            t = np.linspace(0, 10, self.seq_len)
            feature = np.zeros((self.seq_len, self.feature_dim))

            for dim in range(self.feature_dim):
                freq = base_freq + (dim % 10) * 0.01
                phase = np.random.rand() * 2 * np.pi
                feature[:, dim] = (
                    np.sin(2 * np.pi * freq * t + phase) +
                    0.3 * np.sin(4 * np.pi * freq * t + phase) +
                    0.1 * np.random.randn(self.seq_len)
                )

            feature = feature / (np.std(feature) + 1e-8)
            separated_features.append(feature)

        # 混合特徵
        mixed_features = np.sum(separated_features, axis=0)

        # Padding
        while len(separated_features) < self.max_speakers:
            separated_features.append(np.zeros((self.seq_len, self.feature_dim)))

        # Speaker mask
        speaker_mask = np.zeros(self.max_speakers)
        speaker_mask[:num_speakers] = 1.0

        return {
            'mixed_features': torch.FloatTensor(mixed_features),
            'separated_features': torch.FloatTensor(np.stack(separated_features, axis=0)),
            'num_speakers': num_speakers,
            'speaker_mask': torch.FloatTensor(speaker_mask)
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def _save_to_disk(self):
        """將數據保存到磁碟"""
        # 創建目錄
        os.makedirs(self.save_dir, exist_ok=True)

        # 保存數據和元信息
        data_to_save = {
            'samples': self.samples,
            'num_samples': self.num_samples,
            'max_speakers': self.max_speakers,
            'seq_len': self.seq_len,
            'feature_dim': self.feature_dim
        }

        torch.save(data_to_save, self.data_path)

    def _load_from_disk(self):
        """從磁碟加載數據"""
        data = torch.load(self.data_path)

        # 驗證元信息
        if data['num_samples'] != self.num_samples:
            print(f"⚠️  警告：文件中的樣本數 ({data['num_samples']}) 與請求的不同 ({self.num_samples})")

        if data['max_speakers'] != self.max_speakers:
            print(f"⚠️  警告：文件中的 max_speakers ({data['max_speakers']}) 與請求的不同 ({self.max_speakers})")

        # 更新實例變量以匹配加載的數據
        self.num_samples = data['num_samples']
        self.max_speakers = data['max_speakers']
        self.seq_len = data['seq_len']
        self.feature_dim = data['feature_dim']

        return data['samples']


def collate_fn(batch):
    """整理 batch"""
    return {
        'mixed_features': torch.stack([s['mixed_features'] for s in batch]),
        'separated_features': torch.stack([s['separated_features'] for s in batch]),
        'num_speakers': torch.LongTensor([s['num_speakers'] for s in batch]),
        'speaker_mask': torch.stack([s['speaker_mask'] for s in batch])
    }


class MultiSpeakerDataset(Dataset):
    """
    多人語音分離數據集（從 multi_speaker_data 加載）

    使用真實的數據文件（.npy 格式）進行訓練
    """

    def __init__(self, data_dir='multi_speaker_data', split='train', max_speakers=3):
        """
        Args:
            data_dir: 數據目錄
            split: 'train' 或 'val'
            max_speakers: 最大說話者數量
        """
        self.data_dir = os.path.join(data_dir, split)
        self.max_speakers = max_speakers

        # 讀取 metadata
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"✓ 加載 {split} 數據集: {len(self.metadata)} 個樣本")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata[idx]

        # 讀取混合特徵
        mixed_path = os.path.join(self.data_dir, sample['mixed_audio'])
        mixed = np.load(mixed_path)

        # 讀取分離特徵
        separated_list = []
        for speaker in sample['speakers']:
            sep_path = os.path.join(self.data_dir, speaker['feature_file'])
            sep = np.load(sep_path)
            separated_list.append(sep)

        # Padding 到 max_speakers
        num_speakers = len(separated_list)
        while len(separated_list) < self.max_speakers:
            separated_list.append(np.zeros_like(separated_list[0]))

        # Speaker mask
        speaker_mask = np.zeros(self.max_speakers)
        speaker_mask[:num_speakers] = 1.0

        return {
            'mixed_features': torch.FloatTensor(mixed),
            'separated_features': torch.FloatTensor(np.stack(separated_list, axis=0)),
            'num_speakers': num_speakers,
            'speaker_mask': torch.FloatTensor(speaker_mask)
        }


def compute_separation_loss(separated, gt_separated, speaker_mask):
    """計算分離損失"""
    loss = 0
    num_active = 0

    for b in range(separated.shape[0]):
        for s in range(separated.shape[1]):
            if speaker_mask[b, s] > 0.5:
                loss += F.mse_loss(separated[b, s], gt_separated[b, s])
                num_active += 1

    return loss / max(num_active, 1)


def train_speaker_separator_smoke_test(
    max_speakers=3,
    num_epochs=2,
    batch_size=4,
    learning_rate=1e-4,
    device='mps',
    load_from_disk=False
):
    """
    執行訓練 Smoke Test

    快速驗證訓練流程能正常運行

    Args:
        max_speakers: 最大說話者數量
        num_epochs: 訓練輪數
        batch_size: Batch 大小
        learning_rate: 學習率
        device: 使用的設備 ('mps', 'cuda', 'cpu')
        load_from_disk: 是否從磁碟加載數據（False=生成新數據，True=加載已保存數據）
    """
    print("\n" + "="*100)
    print("  Speaker Separator 訓練 Smoke Test")
    print("="*100)

    # 創建數據
    print("\n[1/5] 創建合成數據集...")
    train_dataset = SyntheticMultiSpeakerDataset(
        num_samples=30,
        max_speakers=max_speakers,
        seq_len=100,
        feature_dim=1280,
        seed=42,
        dataset_name='synthetic_train',
        load_from_disk=load_from_disk
    )
    val_dataset = SyntheticMultiSpeakerDataset(
        num_samples=10,
        max_speakers=max_speakers,
        seq_len=100,
        feature_dim=1280,
        seed=999,
        dataset_name='synthetic_val',
        load_from_disk=load_from_disk
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"✓ 訓練: {len(train_dataset)} 樣本, 驗證: {len(val_dataset)} 樣本")

    # 創建模型
    print("\n[2/5] 創建模型...")
    model = SpeakerSeparatorTransformer(
        feature_dim=1280,
        max_speakers=max_speakers,
        num_layers=2,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1
    ).to(device)

    print(f"✓ 參數量: {model.get_num_params():,}")

    # 優化器
    print("\n[3/5] 設置優化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    print("✓ AdamW 已設置")

    # 驗證前向傳播
    print("\n[4/5] 驗證前向傳播...")
    test_batch = next(iter(train_loader))
    test_mixed = test_batch['mixed_features'].to(device)

    with torch.no_grad():
        test_separated, test_probs = model(test_mixed)
    print(f"✓ 輸入: {test_mixed.shape}, 輸出: {test_separated.shape}")

    # 訓練循環
    print("\n[5/5] 開始訓練...")
    print("-" * 100)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            mixed = batch['mixed_features'].to(device)
            gt_separated = batch['separated_features'].to(device)
            speaker_mask = batch['speaker_mask'].to(device)

            # 前向
            separated, probs = model(mixed)

            # 損失
            sep_loss = compute_separation_loss(separated, gt_separated, speaker_mask)

            # 重建損失
            reconstructed = torch.sum(
                separated * speaker_mask.unsqueeze(-1).unsqueeze(-1),
                dim=1
            )
            rec_loss = F.mse_loss(reconstructed, mixed)

            # Activity 損失
            act_loss = F.binary_cross_entropy(probs, speaker_mask)

            # 總損失
            loss = sep_loss + 0.5 * rec_loss + 0.5 * act_loss

            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if (batch_idx + 1) % 2 == 0 or batch_idx == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Sep: {sep_loss.item():.4f}, "
                      f"Rec: {rec_loss.item():.4f}, Act: {act_loss.item():.4f}")

        # 驗證
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                mixed = batch['mixed_features'].to(device)
                gt_separated = batch['separated_features'].to(device)
                speaker_mask = batch['speaker_mask'].to(device)

                separated, probs = model(mixed)

                sep_loss = compute_separation_loss(separated, gt_separated, speaker_mask)
                reconstructed = torch.sum(
                    separated * speaker_mask.unsqueeze(-1).unsqueeze(-1),
                    dim=1
                )
                rec_loss = F.mse_loss(reconstructed, mixed)
                act_loss = F.binary_cross_entropy(probs, speaker_mask)

                loss = sep_loss + 0.5 * rec_loss + 0.5 * act_loss
                val_losses.append(loss.item())

        epoch_time = time.time() - epoch_start
        print("-" * 100)
        print(f"Epoch {epoch+1} 完成 ({epoch_time:.2f}s) - "
              f"訓練: {np.mean(train_losses):.4f}, 驗證: {np.mean(val_losses):.4f}")
        print("-" * 100)

    # 測試
    print("\n[測試] 分離效果...")
    model.eval()
    test_sample = val_dataset[0]
    mixed = test_sample['mixed_features'].unsqueeze(0).to(device)
    num_speakers = test_sample['num_speakers']

    with torch.no_grad():
        separated, probs = model(mixed)

    probs = probs.squeeze(0).cpu()
    print(f"\n實際說話者: {num_speakers}")
    print(f"預測概率:")
    for i in range(len(probs)):
        print(f"  Speaker {i+1}: {probs[i].item():.4f} {'✓' if probs[i] > 0.5 else ''}")

    print("\n" + "="*100)
    print("  ✅ Smoke Test 完成！所有流程正常運行")
    print("="*100)

    return model


def train_speaker_separator(
    data_dir='multi_speaker_data',
    max_speakers=3,
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    device='mps',
    checkpoint_dir='checkpoints',
    save_every=5
):
    """
    使用真實數據訓練 Speaker Separator

    Args:
        data_dir: 數據目錄（包含 train/ 和 val/）
        max_speakers: 最大說話者數量
        num_epochs: 訓練輪數
        batch_size: Batch 大小
        learning_rate: 學習率
        device: 使用的設備 ('mps', 'cuda', 'cpu')
        checkpoint_dir: Checkpoint 保存目錄
        save_every: 每幾個 epoch 保存一次 checkpoint
    """
    print("\n" + "="*100)
    print("  Speaker Separator 完整訓練")
    print("="*100)
    print(f"\n配置:")
    print(f"  數據目錄: {data_dir}")
    print(f"  最大說話者數: {max_speakers}")
    print(f"  訓練輪數: {num_epochs}")
    print(f"  Batch 大小: {batch_size}")
    print(f"  學習率: {learning_rate}")
    print(f"  設備: {device}")
    print(f"  Checkpoint 目錄: {checkpoint_dir}")

    # 創建 checkpoint 目錄
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 載入數據
    print("\n[1/6] 載入數據集...")
    train_dataset = MultiSpeakerDataset(data_dir, split='train', max_speakers=max_speakers)
    val_dataset = MultiSpeakerDataset(data_dir, split='val', max_speakers=max_speakers)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"✓ 訓練集: {len(train_dataset)} 個樣本 ({len(train_loader)} batches)")
    print(f"✓ 驗證集: {len(val_dataset)} 個樣本 ({len(val_loader)} batches)")

    # 創建模型
    print("\n[2/6] 創建模型...")
    model = SpeakerSeparatorTransformer(
        feature_dim=1280,
        max_speakers=max_speakers,
        num_layers=4,  # 使用更深的模型
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    ).to(device)

    num_params = model.get_num_params()
    print(f"✓ 參數量: {num_params:,}")

    # 優化器和學習率調度
    print("\n[3/6] 設置優化器和調度器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Cosine annealing 學習率調度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.1
    )

    print("✓ AdamW 優化器已設置")
    print("✓ CosineAnnealingLR 調度器已設置")

    # 驗證前向傳播
    print("\n[4/6] 驗證前向傳播...")
    test_batch = next(iter(train_loader))
    test_mixed = test_batch['mixed_features'].to(device)

    with torch.no_grad():
        test_separated, test_probs = model(test_mixed)
    print(f"✓ 輸入形狀: {test_mixed.shape}")
    print(f"✓ 輸出形狀: {test_separated.shape}")

    # 訓練歷史
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')

    # 訓練循環
    print("\n[5/6] 開始訓練...")
    print("="*100)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 訓練階段
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            mixed = batch['mixed_features'].to(device)
            gt_separated = batch['separated_features'].to(device)
            speaker_mask = batch['speaker_mask'].to(device)

            # 前向傳播
            separated, probs = model(mixed)

            # 計算損失
            sep_loss = compute_separation_loss(separated, gt_separated, speaker_mask)

            reconstructed = torch.sum(
                separated * speaker_mask.unsqueeze(-1).unsqueeze(-1),
                dim=1
            )
            rec_loss = F.mse_loss(reconstructed, mixed)

            act_loss = F.binary_cross_entropy(probs, speaker_mask)

            # 總損失
            loss = sep_loss + 0.5 * rec_loss + 0.5 * act_loss

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # 定期打印
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Sep: {sep_loss.item():.4f}, "
                      f"Rec: {rec_loss.item():.4f}, "
                      f"Act: {act_loss.item():.4f}")

        # 驗證階段
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                mixed = batch['mixed_features'].to(device)
                gt_separated = batch['separated_features'].to(device)
                speaker_mask = batch['speaker_mask'].to(device)

                separated, probs = model(mixed)

                sep_loss = compute_separation_loss(separated, gt_separated, speaker_mask)
                reconstructed = torch.sum(
                    separated * speaker_mask.unsqueeze(-1).unsqueeze(-1),
                    dim=1
                )
                rec_loss = F.mse_loss(reconstructed, mixed)
                act_loss = F.binary_cross_entropy(probs, speaker_mask)

                loss = sep_loss + 0.5 * rec_loss + 0.5 * act_loss
                val_losses.append(loss.item())

        # 計算平均損失
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        # 記錄歷史
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['learning_rate'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print("-" * 100)
        print(f"Epoch {epoch+1}/{num_epochs} 完成 ({epoch_time:.2f}s) - "
              f"訓練: {avg_train_loss:.4f}, 驗證: {avg_val_loss:.4f}, "
              f"LR: {current_lr:.6f}")
        print("-" * 100)

        # 更新學習率
        scheduler.step()

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_history': train_history
            }, best_checkpoint_path)
            print(f"✓ 保存最佳模型到: {best_checkpoint_path}")

        # 定期保存 checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_history': train_history
            }, checkpoint_path)
            print(f"✓ 保存 checkpoint 到: {checkpoint_path}")

    print("\n" + "="*100)
    print("  ✅ 訓練完成！")
    print("="*100)
    print(f"\n最佳驗證損失: {best_val_loss:.4f}")
    print(f"最佳模型: {best_checkpoint_path}")

    return model, train_history


# ============================================================================
# 第五部分：主程式（Demo）
# ============================================================================

def main():
    """
    Demo：展示如何使用 Speaker Separator

    選項：
    1. smoke_test - 執行訓練 smoke test（快速驗證）
    2. train - 使用真實數據訓練模型
    3. (默認) - 加載完整 Qwen2-Audio 並添加 separator
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║      在 Qwen2-Audio 架構內實現多人語音分離                                ║
║                                                                          ║
║      核心創新：                                                          ║
║        • 在 Whisper 特徵層進行分離（不是音頻信號層）                     ║
║        • 善用 Whisper 的預訓練表示能力                                   ║
║        • 使用 Speaker Queries + Cross-Attention                        ║
║        • 完全整合在 Qwen2-Audio 架構中                                   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n使用設備: {device}")

    # 選擇模式
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == 'smoke_test':
            # 執行 smoke test
            print("\n執行訓練 Smoke Test...")
            load_from_disk = '--load' in sys.argv

            model = train_speaker_separator_smoke_test(
                max_speakers=3,
                num_epochs=2,
                batch_size=4,
                learning_rate=1e-4,
                device=device,
                load_from_disk=load_from_disk
            )
            return

        elif mode == 'train':
            # 執行完整訓練
            print("\n執行完整訓練...")

            # 解析訓練參數
            num_epochs = 10
            batch_size = 8
            learning_rate = 1e-4

            for arg in sys.argv[2:]:
                if arg.startswith('--epochs='):
                    num_epochs = int(arg.split('=')[1])
                elif arg.startswith('--batch_size='):
                    batch_size = int(arg.split('=')[1])
                elif arg.startswith('--lr='):
                    learning_rate = float(arg.split('=')[1])

            model, history = train_speaker_separator(
                data_dir='multi_speaker_data',
                max_speakers=3,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                checkpoint_dir='checkpoints',
                save_every=5
            )
            return

    # 否則執行原本的 demo（加載完整模型）
    model_name = "Qwen/Qwen2-Audio-7B-Instruct"

    # ========================================================================
    # 步驟 1: 加載模型
    # ========================================================================

    print("\n加載 Qwen2-Audio...")

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    print("✓ 模型加載完成")

    # ========================================================================
    # 步驟 2: 添加 Speaker Separator
    # ========================================================================

    model = add_speaker_separator_to_qwen2audio(
        model,
        max_speakers=4,
        num_layers=4,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    )

    # ========================================================================
    # 步驟 3: 驗證設置
    # ========================================================================

    print("\n驗證：")
    print(f"  ✓ speaker_separator 已添加: {hasattr(model, 'speaker_separator')}")
    print(f"  ✓ audio_tower forward 已修改")

    # 打印架構
    print("\n完整架構：")
    print("  輸入：混合音頻（多人同時說話）")
    print("    ↓")
    print("  audio_tower (Whisper Encoder) [凍結]")
    print("    → 提取混合特徵 [seq_len, 1280]")
    print("    ↓")
    print("  Speaker Separator [訓練]")
    print("    → 分離成 N 個說話者 [N, seq_len, 1280]")
    print("    ↓")
    print("  N × projector [凍結，重複使用]")
    print("    → N × embeddings [seq_len, 4096]")
    print("    ↓")
    print("  N × LLM [凍結]")
    print("    → N × 轉錄文本")

    print("\n關鍵優勢：")
    print("  1. 利用 Whisper 的強大特徵表示")
    print("  2. 在語義空間分離（比音頻信號層更高效）")
    print("  3. 重複使用預訓練的 projector 和 LLM")
    print("  4. 完全在 Qwen2-Audio 架構內部")
    print("  5. 不需要外部語音分離模型")

    print("\n下一步：")
    print("  1. 準備訓練數據（混合音頻 + 分離標註）")
    print("  2. 實現 PIT loss 訓練")
    print("  3. 測試分離效果")
    print("  4. 整合轉錄流程")

    print("="*100)


if __name__ == "__main__":
    main()
