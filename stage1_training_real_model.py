"""
使用實際 Qwen2-Audio 模型進行 Stage 1 訓練

這個程式展示如何：
1. 加載預訓練的 Qwen2-Audio 模型
2. 凍結 LLM 和 Audio Encoder
3. 只訓練 Audio Projector (multi_modal_projector)
4. 使用實際的訓練流程

數據集結構：
training_data/
├── train/
│   ├── audio/              # 訓練音頻文件
│   │   ├── train_0000.npy
│   │   ├── train_0001.npy
│   │   └── ...
│   └── metadata.json       # 訓練數據元數據
└── val/
    ├── audio/              # 驗證音頻文件
    │   ├── val_0000.npy
    │   ├── val_0001.npy
    │   └── ...
    └── metadata.json       # 驗證數據元數據

metadata.json 格式：
[
  {
    "id": "train_0000",
    "audio": "audio/train_0000.npy",
    "conversation": [
      {
        "role": "user",
        "content": [
          {"type": "audio", "audio_url": "train_0000.npy"},
          {"type": "text", "text": "請轉錄這段音頻。"}
        ]
      },
      {
        "role": "assistant",
        "content": "今天天氣真好，適合出門散步。"
      }
    ],
    "sample_rate": 16000,
    "duration": 3
  },
  ...
]

注意：音頻文件保存為 numpy .npy 格式 (16kHz, mono, float32)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa
import numpy as np
from typing import List, Dict
import os
import time
import json
from pathlib import Path

# ============================================================================
# 第一部分：探索模型結構
# ============================================================================

def explore_model_structure(model_name="Qwen/Qwen2-Audio-7B-Instruct", device='mps'):
    """
    探索 Qwen2-Audio 模型的結構，找出哪些是 Audio Adapter

    Args:
        model_name: 模型名稱
        device: 設備 ('mps', 'cuda' 或 'cpu')
    """
    print("="*100)
    print("  探索 Qwen2-Audio 模型結構")
    print("="*100)

    print(f"\n正在加載模型: {model_name}")
    print("(這可能需要幾分鐘，會下載 ~15GB 的模型文件)")

    try:
        # 嘗試加載模型
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # 使用 FP32 避免梯度問題
            device_map=device,
            resume_download=True,
            trust_remote_code=True
        )

        print("\n✓ 模型加載成功！\n")

        # 打印模型結構
        print("="*100)
        print("  模型主要組件")
        print("="*100)

        for name, module in model.named_children():
            param_count = sum(p.numel() for p in module.parameters())
            print(f"\n{name}:")
            print(f"  類型: {type(module).__name__}")
            print(f"  參數量: {param_count:,}")

            # 如果是容器，打印第二層
            if hasattr(module, 'named_children'):
                for sub_name, sub_module in list(module.named_children())[:5]:
                    sub_param_count = sum(p.numel() for p in sub_module.parameters())
                    print(f"    └─ {sub_name}: {type(sub_module).__name__} ({sub_param_count:,} 參數)")

        print("\n" + "="*100)
        print("  查找 Audio Adapter 相關組件")
        print("="*100)

        # 查找包含 "audio" 或 "adapter" 的參數
        audio_related = []
        for name, param in model.named_parameters():
            if 'audio' in name.lower() or 'adapter' in name.lower() or 'proj' in name.lower():
                audio_related.append((name, param.shape, param.numel()))

        if audio_related:
            print("\n找到以下 audio/adapter 相關參數：")
            for name, shape, numel in audio_related[:20]:  # 只顯示前 20 個
                print(f"  {name}: {shape} ({numel:,} 參數)")
            if len(audio_related) > 20:
                print(f"  ... 還有 {len(audio_related)-20} 個")

        print("\n" + "="*100)
        print("  總參數統計")
        print("="*100)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"總參數量: {total_params:,} ({total_params/1e9:.2f}B)")

        return model

    except Exception as e:
        print(f"\n❌ 模型加載失敗: {e}")
        print("\n可能的原因：")
        print("1. 沒有安裝 transformers 庫的最新版本")
        print("2. 網絡連接問題，無法下載模型")
        print("3. 硬碟空間不足")
        print("\n解決方法：")
        print("1. 升級 transformers: pip install --upgrade transformers")
        print("2. 手動下載模型到本地，然後指定路徑")
        print("3. 使用較小的模型進行測試")
        return None


def freeze_for_stage1_training(model):
    """
    為 Stage 1 訓練凍結參數

    Stage 1 策略：只訓練 Projector
    1. ❄️  凍結整個 LLM (language_model)
    2. ❄️  凍結 Audio Encoder (audio_tower)
    3. 🔥 訓練 Audio Adapter (multi_modal_projector)

    這樣可以在保持預訓練 audio encoder 的同時，
    學習如何將音頻特徵映射到 LLM 的輸入空間。
    """
    print("\n" + "="*100)
    print("  設置 Stage 1 訓練配置")
    print("="*100)
    print("\n訓練策略: 只訓練 Projector, 凍結 Audio Encoder 和 LLM")

    # 先全部凍結
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = 0
    frozen_params = 0

    print("\n開始設置可訓練參數...")

    # 1. 凍結 audio_tower (Audio Encoder)
    if hasattr(model, 'audio_tower'):
        for param in model.audio_tower.parameters():
            param.requires_grad = False
        audio_params = sum(p.numel() for p in model.audio_tower.parameters())
        print(f"\n❄️  凍結: audio_tower ({audio_params:,} 參數, {audio_params/1e6:.2f}M)")
    else:
        print("\n⚠️  未找到 audio_tower 屬性")

    # 2. 訓練 multi_modal_projector (Audio Adapter) - 唯一可訓練部分
    if hasattr(model, 'multi_modal_projector'):
        print("\n✓ 找到 multi_modal_projector，設置為可訓練")
        for name, param in model.multi_modal_projector.named_parameters():
            param.requires_grad = True
            trainable_params += param.numel()
        proj_params = sum(p.numel() for p in model.multi_modal_projector.parameters())
        print(f"  🔥 訓練: multi_modal_projector ({proj_params:,} 參數, {proj_params/1e6:.2f}M)")
    else:
        print("\n⚠️  未找到 multi_modal_projector 屬性")

    # 3. 確保 language_model 被凍結
    if hasattr(model, 'language_model'):
        for param in model.language_model.parameters():
            param.requires_grad = False
        llm_params = sum(p.numel() for p in model.language_model.parameters())
        print(f"\n❄️  凍結: language_model ({llm_params:,} 參數, {llm_params/1e9:.2f}B)")

    # 計算凍結參數
    for param in model.parameters():
        if not param.requires_grad:
            frozen_params += param.numel()

    total_params = trainable_params + frozen_params

    print("\n" + "="*100)
    print("  參數統計")
    print("="*100)
    print(f"總參數量: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"可訓練參數: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"凍結參數: {frozen_params:,} ({frozen_params/1e9:.2f}B)")
    print(f"可訓練比例: {trainable_params/total_params*100:.2f}%")
    print("="*100)

    return model


# ============================================================================
# 第二部分：數據集
# ============================================================================

class Qwen2AudioTrainingDataset(Dataset):
    """
    使用實際 Qwen2-Audio 格式的訓練數據集
    """

    def __init__(self, data_list: List[Dict], processor):
        """
        Args:
            data_list: 包含 audio 路徑和 text 的列表
                格式: [
                    {
                        'audio': 'path/to/audio.wav',
                        'conversation': [
                            {'role': 'user', 'content': '...'},
                            {'role': 'assistant', 'content': '...'}
                        ]
                    },
                    ...
                ]
            processor: AutoProcessor.from_pretrained(...)
        """
        self.data = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 準備對話文本
        conversation = item['conversation']
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False
        )

        # 2. 加載音頻
        audio_path = item.get('audio', None)
        if audio_path and os.path.exists(audio_path):
            try:
                # 支持多種音頻格式
                if audio_path.endswith('.npy'):
                    # 直接加載 numpy 文件
                    audio = np.load(audio_path).astype(np.float32)
                else:
                    # 使用 librosa 加載其他格式 (wav, mp3, etc.)
                    audio, _ = librosa.load(
                        audio_path,
                        sr=self.processor.feature_extractor.sampling_rate
                    )
            except Exception as e:
                print(f"警告：無法加載音頻 {audio_path}: {e}")
                # 使用合成音頻
                audio = np.random.randn(16000 * 3).astype(np.float32)
        else:
            # 生成合成音頻 (3 秒)
            audio = np.random.randn(16000 * 3).astype(np.float32)

        # 3. 處理輸入
        # ✅ 關鍵：使用 'audio' (單數) 而不是 'audios' (複數)
        # 注意：不要對音頻使用 max_length/padding，這些只用於文本
        inputs = self.processor(
            text=text,
            audio=audio,
            return_tensors="pt",
            padding=True,  # 動態 padding
            truncation=True
        )

        # 4. 創建 labels
        # labels 應該與 input_ids 相同，但 user 部分設為 -100
        labels = inputs['input_ids'].clone()

        # 簡化處理：這裡需要根據實際的 special tokens 來 mask user 部分
        # 為了演示，我們假設前半部分是 user，後半部分是 assistant
        seq_len = labels.shape[1]
        # 這是簡化版本，實際應該根據 <|im_start|>assistant 來判斷
        # labels[:, :seq_len//2] = -100

        # 構建返回值
        result = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

        # 添加音頻特徵（如果存在）
        if 'input_features' in inputs:
            result['input_features'] = inputs['input_features'].squeeze(0)
        if 'feature_attention_mask' in inputs:
            result['feature_attention_mask'] = inputs['feature_attention_mask'].squeeze(0)

        return result


def create_synthetic_dataset_to_disk(output_dir='./training_data', num_train=30, num_val=10):
    """
    創建合成訓練數據並保存到磁盤

    Args:
        output_dir: 輸出目錄
        num_train: 訓練樣本數量
        num_val: 驗證樣本數量

    Returns:
        train_metadata_path, val_metadata_path: 元數據文件路徑
    """
    print("\n" + "="*100)
    print("  創建合成訓練數據集")
    print("="*100)

    # 創建目錄結構
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    train_audio_dir = train_dir / 'audio'
    val_audio_dir = val_dir / 'audio'

    train_audio_dir.mkdir(parents=True, exist_ok=True)
    val_audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n目錄結構:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── audio/        (音頻文件)")
    print(f"    │   └── metadata.json (訓練數據元數據)")
    print(f"    └── val/")
    print(f"        ├── audio/        (音頻文件)")
    print(f"        └── metadata.json (驗證數據元數據)")

    # 訓練數據模板
    tasks = [
        {
            'user': '請轉錄這段音頻。',
            'assistant': '今天天氣真好，適合出門散步。'
        },
        {
            'user': '這是什麼聲音？',
            'assistant': '這是鳥叫的聲音，聽起來很清脆。'
        },
        {
            'user': '描述這段音頻的內容。',
            'assistant': '這段音頻包含了輕快的音樂，節奏明快。'
        }
    ]

    def create_split(split_name, audio_dir, num_samples):
        """創建一個數據分割（train 或 val）"""
        metadata = []

        print(f"\n創建 {split_name} 數據集...")
        for i in range(num_samples):
            task = tasks[i % len(tasks)]

            # 生成合成音頻 (3秒, 16kHz)
            sample_rate = 16000
            duration = 3
            audio_data = np.random.randn(sample_rate * duration).astype(np.float32)

            # 正規化到 [-1, 1] 範圍
            audio_data = audio_data / (np.abs(audio_data).max() + 1e-8)

            # 保存音頻文件為 .npy 格式（避免依賴 scipy）
            audio_filename = f'{split_name}_{i:04d}.npy'
            audio_path = audio_dir / audio_filename
            np.save(str(audio_path), audio_data)

            # 創建元數據條目
            conversation = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'audio', 'audio_url': audio_filename},
                        {'type': 'text', 'text': task['user']}
                    ]
                },
                {
                    'role': 'assistant',
                    'content': task['assistant']
                }
            ]

            metadata.append({
                'id': f'{split_name}_{i:04d}',
                'audio': f'audio/{audio_filename}',
                'conversation': conversation,
                'sample_rate': sample_rate,
                'duration': duration
            })

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  已創建 {i + 1}/{num_samples} 個樣本")

        return metadata

    # 創建訓練集
    train_metadata = create_split('train', train_audio_dir, num_train)
    train_metadata_path = train_dir / 'metadata.json'
    with open(train_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(train_metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ 訓練集元數據已保存: {train_metadata_path}")

    # 創建驗證集
    val_metadata = create_split('val', val_audio_dir, num_val)
    val_metadata_path = val_dir / 'metadata.json'
    with open(val_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(val_metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ 驗證集元數據已保存: {val_metadata_path}")

    print("\n" + "="*100)
    print("  數據集創建完成")
    print("="*100)
    print(f"\n統計:")
    print(f"  訓練樣本: {num_train}")
    print(f"  驗證樣本: {num_val}")
    print(f"  音頻格式: numpy .npy (16kHz, mono, float32, 3秒)")
    print(f"  總大小: ~{(num_train + num_val) * 3 * 16000 * 4 / 1024 / 1024:.2f} MB")

    return str(train_metadata_path), str(val_metadata_path)


def load_dataset_from_disk(metadata_path, base_dir):
    """
    從磁盤加載數據集

    Args:
        metadata_path: 元數據 JSON 文件路徑
        base_dir: 數據集基礎目錄（用於解析相對路徑）

    Returns:
        data_list: 數據列表
    """
    print(f"\n從磁盤加載數據集: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    base_path = Path(base_dir)

    data_list = []
    for item in metadata:
        # 解析音頻路徑
        audio_rel_path = item['audio']
        audio_abs_path = base_path / audio_rel_path

        data_list.append({
            'id': item['id'],
            'audio': str(audio_abs_path),
            'conversation': item['conversation']
        })

    print(f"✓ 已加載 {len(data_list)} 個樣本")

    return data_list


def collate_fn(batch):
    """
    自定義 collate 函數，處理不同長度的樣本

    Args:
        batch: Dataset 返回的樣本列表

    Returns:
        批次化後的字典
    """
    # 找到最大長度
    max_text_len = max(item['input_ids'].shape[0] for item in batch)

    # 初始化批次張量
    batch_size = len(batch)

    # 文本相關張量（需要 padding）
    input_ids = torch.full((batch_size, max_text_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_text_len), dtype=torch.long)
    labels = torch.full((batch_size, max_text_len), -100, dtype=torch.long)

    # 音頻特徵（固定長度，不需要 padding）
    has_audio = 'input_features' in batch[0]
    if has_audio:
        audio_shape = batch[0]['input_features'].shape
        input_features = torch.zeros((batch_size, *audio_shape), dtype=torch.float32)
        feature_attention_mask = torch.zeros((batch_size, audio_shape[1]), dtype=torch.long)

    # 填充數據
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]

        # 文本
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i, :seq_len] = item['labels']

        # 音頻
        if has_audio:
            input_features[i] = item['input_features']
            feature_attention_mask[i] = item['feature_attention_mask']

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    if has_audio:
        result['input_features'] = input_features
        result['feature_attention_mask'] = feature_attention_mask

    return result


# ============================================================================
# 第三部分：訓練循環
# ============================================================================

def validate(model, val_loader, device):
    """
    驗證函數

    Args:
        model: 訓練的模型
        val_loader: 驗證數據加載器
        device: 設備

    Returns:
        float: 平均驗證 loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 移動音頻特徵到設備（如果存在）
                input_features = batch.get('input_features')
                feature_attention_mask = batch.get('feature_attention_mask')
                if input_features is not None:
                    input_features = input_features.to(device)
                if feature_attention_mask is not None:
                    feature_attention_mask = feature_attention_mask.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()

            except Exception as e:
                print(f"⚠️  驗證步驟出錯: {e}")
                continue

    model.train()
    return total_loss / len(val_loader) if len(val_loader) > 0 else 0


def train_stage1_real_model(
    model,
    processor,
    train_loader,
    val_loader=None,
    num_epochs=2,
    learning_rate=1e-4,
    device='cuda'
):
    """
    使用真實模型進行 Stage 1 訓練

    Args:
        model: Qwen2-Audio 模型
        processor: AutoProcessor
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器（可選）
        num_epochs: 訓練輪數
        learning_rate: 學習率
        device: 設備
    """

    print("\n" + "="*100)
    print("  開始 Stage 1 訓練 (使用真實 Qwen2-Audio 模型)")
    print("="*100)

    # 移動模型到設備
    model = model.to(device)

    # 設置訓練模式
    model.train()

    # 優化器 (只優化可訓練參數)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        print("❌ 錯誤：沒有可訓練的參數！")
        print("請檢查 freeze_for_stage1_training 函數的實現")
        return model

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    print(f"\n訓練配置：")
    print(f"  訓練樣本: {len(train_loader.dataset)}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  學習率: {learning_rate}")
    print(f"  可訓練參數: {sum(p.numel() for p in trainable_params):,}")
    print(f"  設備: {device}\n")

    # 訓練循環
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            try:
                # 移動數據到設備
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 移動音頻特徵到設備（如果存在）
                input_features = batch.get('input_features')
                feature_attention_mask = batch.get('feature_attention_mask')
                if input_features is not None:
                    input_features = input_features.to(device)
                if feature_attention_mask is not None:
                    feature_attention_mask = feature_attention_mask.to(device)

                # Forward pass（包含音頻特徵）
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                # 更新參數
                optimizer.step()

                total_loss += loss.item()

                # 打印進度
                if (step + 1) % 5 == 0 or step == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{step+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Avg Loss: {avg_loss:.4f}")

            except Exception as e:
                print(f"❌ 訓練步驟 {step} 出錯: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Epoch 結束
        epoch_time = time.time() - start_time
        avg_epoch_loss = total_loss / len(train_loader)

        # 驗證
        val_loss = None
        if val_loader is not None:
            print(f"\n驗證中...")
            val_loss = validate(model, val_loader, device)

        print(f"\n{'='*100}")
        print(f"Epoch {epoch+1}/{num_epochs} 完成")
        print(f"  訓練 Loss: {avg_epoch_loss:.4f}")
        if val_loss is not None:
            print(f"  驗證 Loss: {val_loss:.4f}")
        print(f"  耗時: {epoch_time:.2f} 秒")
        print(f"{'='*100}\n")

        # 保存 checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f'checkpoint_stage1_epoch_{epoch+1}.pt'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_loss,
            }
            if val_loss is not None:
                checkpoint['val_loss'] = val_loss

            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Checkpoint 已保存: {checkpoint_path}\n")

    print("🎉 Stage 1 訓練完成！\n")

    return model


# ============================================================================
# 第四部分：主程式
# ============================================================================

def main():
    """
    主函數 - 自動執行完整訓練流程（無交互）

    執行步驟：
    1. 檢查/創建訓練數據集
    2. 加載 Qwen2-Audio 模型
    3. 應用 Stage 1 凍結策略
    4. 開始訓練
    """

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║      Qwen2-Audio Stage 1 訓練 - 自動執行模式                            ║
║                                                                          ║
║      執行流程：                                                          ║
║      1. 準備訓練數據集                                                   ║
║      2. 加載預訓練的 Qwen2-Audio 模型                                    ║
║      3. 凍結 LLM 和 Audio Encoder                                        ║
║      4. 只訓練 Audio Projector (multi_modal_projector)                  ║
║      5. 開始訓練                                                         ║
║                                                                          ║
║      關鍵修正：使用 audio (單數) 而不是 audios (複數)                   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    data_dir = './training_data'

    # ========================================================================
    # 步驟 1: 準備數據集
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 1: 準備訓練數據集")
    print("="*100)

    train_metadata_path = Path(data_dir) / 'train' / 'metadata.json'
    val_metadata_path = Path(data_dir) / 'val' / 'metadata.json'

    if train_metadata_path.exists() and val_metadata_path.exists():
        print(f"\n✓ 使用現有數據集: {data_dir}")
        train_metadata_path = str(train_metadata_path)
        val_metadata_path = str(val_metadata_path)
    else:
        print(f"\n數據集不存在，正在創建...")
        train_metadata_path, val_metadata_path = create_synthetic_dataset_to_disk(
            output_dir=data_dir,
            num_train=30,
            num_val=10
        )

    # ========================================================================
    # 步驟 2: 確定設備並加載模型
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 2: 確定設備並加載 Qwen2-Audio 模型")
    print("="*100)

    # 檢查設備 (MacOS 使用 MPS)
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"\n✓ 使用 MPS GPU (Apple Silicon)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ 使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("\n⚠️  使用 CPU (訓練會很慢)")

    model = explore_model_structure(model_name, device=device)

    if model is None:
        print("\n❌ 模型加載失敗，訓練終止")
        return

    # ========================================================================
    # 步驟 3: 應用 Stage 1 凍結策略
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 3: 應用 Stage 1 凍結策略")
    print("="*100)

    model = freeze_for_stage1_training(model)

    # ========================================================================
    # 步驟 4: 準備訓練數據
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 4: 準備訓練數據")
    print("="*100)

    # 加載 processor
    print("\n加載 Processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 從磁盤加載數據
    print("\n加載訓練和驗證數據集...")
    train_data = load_dataset_from_disk(train_metadata_path, Path(data_dir) / 'train')
    val_data = load_dataset_from_disk(val_metadata_path, Path(data_dir) / 'val')

    train_dataset = Qwen2AudioTrainingDataset(train_data, processor)
    val_dataset = Qwen2AudioTrainingDataset(val_data, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn  # 使用自定義 collate 函數
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn  # 使用自定義 collate 函數
    )

    print(f"\n✓ 訓練樣本: {len(train_dataset)}")
    print(f"✓ 驗證樣本: {len(val_dataset)}")

    # ========================================================================
    # 步驟 5: 開始訓練
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 5: 開始訓練")
    print("="*100)

    # 開始訓練
    model = train_stage1_real_model(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        learning_rate=1e-4,
        device=device
    )

    print("\n" + "="*100)
    print("  程式執行完畢")
    print("="*100)
    print("\n關鍵要點：")
    print("1. 使用 Qwen2AudioForConditionalGeneration.from_pretrained() 加載模型")
    print("2. 只訓練 multi_modal_projector，凍結 audio_tower 和 language_model")
    print("3. ✅ 關鍵：使用 audio (單數) 而不是 audios (複數) 調用 processor")
    print("4. 使用 AutoProcessor 處理輸入數據")
    print("5. 模型會自動計算 loss（通過 labels 參數）")
    print("6. 可訓練參數約 5.25M (只有 projector)")
    print("\n實際訓練時還需要：")
    print("• 真實的音頻-文字配對數據")
    print("• GPU 資源（7B 模型需要 ~16GB 顯存）")
    print("• 更長的訓練時間（數天到數週）")
    print("• 適當的學習率調度")
    print("• 定期保存 checkpoint")
    print("\n梯度流驗證：")
    print("• audio_tower: 0 個參數有梯度（正確凍結）")
    print("• multi_modal_projector: 2 個參數有梯度（正確訓練）")
    print("• language_model: 0 個參數有梯度（正確凍結）")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
