"""
使用 LoRA 微調 Qwen2-Audio 模型

這個程式展示正確的微調方式：
1. ✅ 凍結所有預訓練參數（包括 multi_modal_projector）
2. ✅ 只在 LLM 層添加 LoRA adapters
3. ✅ 保持音頻-文本對齊不被破壞
4. ✅ 參數高效訓練

對比：
- ❌ stage1_training_real_model.py: 訓練 projector → 破壞對齊
- ✅ 本腳本: 使用 LoRA → 保留對齊，只調整 LLM 行為

依賴安裝：
pip install peft  # Parameter-Efficient Fine-Tuning library
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import librosa
import numpy as np
from typing import List, Dict
import os
import time
import json
from pathlib import Path

# ============================================================================
# 第一部分：LoRA 配置
# ============================================================================

def setup_lora_model(model, lora_rank=8, lora_alpha=16, target_modules=None):
    """
    為 Qwen2-Audio 設置 LoRA

    LoRA (Low-Rank Adaptation) 原理：
    - 不修改原始權重 W
    - 添加低秩分解：W' = W + BA（其中 B 和 A 是小矩陣）
    - 只訓練 B 和 A，參數量極小

    Args:
        model: Qwen2-Audio 模型
        lora_rank: LoRA 秩（越小參數越少，通常 4-16）
        lora_alpha: LoRA 縮放因子（通常 = 2 * rank）
        target_modules: 要應用 LoRA 的模組名稱

    Returns:
        配置好 LoRA 的模型
    """
    print("\n" + "="*100)
    print("  設置 LoRA 配置")
    print("="*100)

    # 🔧 關鍵修正：在應用 LoRA 之前，先手動凍結 audio_tower 和 multi_modal_projector
    print("\n步驟 1: 凍結 audio_tower 和 multi_modal_projector")

    if hasattr(model, 'audio_tower'):
        for param in model.audio_tower.parameters():
            param.requires_grad = False
        print("  ✓ audio_tower 已凍結")

    if hasattr(model, 'multi_modal_projector'):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False
        print("  ✓ multi_modal_projector 已凍結")

    # 如果沒有指定目標模組，使用默認值（針對 Qwen2 的 attention 層）
    if target_modules is None:
        target_modules = [
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection
        ]

    print(f"\n步驟 2: 配置 LoRA")
    print(f"  Rank (r): {lora_rank}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  目標模組: {target_modules}")
    print(f"  這些模組位於: language_model.model.layers[*].self_attn.*")

    # 創建 LoRA 配置
    lora_config = LoraConfig(
        r=lora_rank,                    # LoRA 秩
        lora_alpha=lora_alpha,          # LoRA 縮放
        target_modules=target_modules,  # 目標模組
        lora_dropout=0.05,              # Dropout
        bias="none",                    # 不訓練 bias
        task_type=TaskType.CAUSAL_LM,   # 任務類型
        modules_to_save=[]              # 不額外保存任何模組
    )

    # 應用 LoRA
    print("\n步驟 3: 應用 LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # 🔧 再次確保 audio_tower 和 multi_modal_projector 被凍結
    print("\n步驟 4: 再次確認凍結狀態...")
    if hasattr(model, 'base_model'):
        # PEFT 包裝後，原始模型在 base_model.model 中
        base = model.base_model.model
        if hasattr(base, 'audio_tower'):
            for param in base.audio_tower.parameters():
                param.requires_grad = False
        if hasattr(base, 'multi_modal_projector'):
            for param in base.multi_modal_projector.parameters():
                param.requires_grad = False
        print("  ✓ 已確認凍結狀態")

    # 打印參數統計
    print("\n" + "="*100)
    print("  LoRA 參數統計")
    print("="*100)

    model.print_trainable_parameters()

    # 詳細統計
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"\n詳細統計:")
    print(f"  總參數: {all_params:,} ({all_params/1e9:.2f}B)")
    print(f"  可訓練參數: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  可訓練比例: {100 * trainable_params / all_params:.4f}%")

    print("\n✅ 關鍵優勢:")
    print("  1. multi_modal_projector 完全凍結 → 音頻-文本對齊保留")
    print("  2. audio_tower 完全凍結 → 音頻編碼能力保留")
    print("  3. language_model 原始權重凍結 → 語言能力保留")
    print("  4. 只訓練 LoRA adapters → 參數高效，不會遺忘")

    print("="*100)

    return model


def verify_lora_setup(model):
    """
    驗證 LoRA 設置是否正確

    檢查：
    1. multi_modal_projector 是否完全凍結
    2. audio_tower 是否完全凍結
    3. 只有 LoRA adapters 可訓練
    """
    print("\n" + "="*100)
    print("  驗證 LoRA 設置")
    print("="*100)

    # 按模組分類參數
    projector_trainable = 0
    projector_frozen = 0
    audio_trainable = 0
    audio_frozen = 0
    lora_trainable = 0
    llm_trainable = 0
    llm_frozen = 0

    for name, param in model.named_parameters():
        if 'multi_modal_projector' in name:
            if param.requires_grad:
                projector_trainable += param.numel()
            else:
                projector_frozen += param.numel()
        elif 'audio_tower' in name:
            if param.requires_grad:
                audio_trainable += param.numel()
            else:
                audio_frozen += param.numel()
        elif 'lora' in name.lower():
            if param.requires_grad:
                lora_trainable += param.numel()
        elif 'language_model' in name:
            if param.requires_grad:
                llm_trainable += param.numel()
            else:
                llm_frozen += param.numel()

    print("\n參數狀態檢查:")

    # multi_modal_projector
    print(f"\n1. multi_modal_projector:")
    print(f"   凍結: {projector_frozen:,} ({projector_frozen/1e6:.2f}M)")
    print(f"   可訓練: {projector_trainable:,}")
    if projector_trainable == 0:
        print("   ✅ 完全凍結 - 音頻-文本對齊保留")
    else:
        print("   ❌ 有參數可訓練 - 會破壞對齊！")

    # audio_tower
    print(f"\n2. audio_tower:")
    print(f"   凍結: {audio_frozen:,} ({audio_frozen/1e6:.2f}M)")
    print(f"   可訓練: {audio_trainable:,}")
    if audio_trainable == 0:
        print("   ✅ 完全凍結 - 音頻編碼能力保留")
    else:
        print("   ❌ 有參數可訓練")

    # language_model (原始權重)
    print(f"\n3. language_model (原始權重):")
    print(f"   凍結: {llm_frozen:,} ({llm_frozen/1e9:.2f}B)")
    print(f"   可訓練: {llm_trainable:,}")
    if llm_trainable == 0:
        print("   ✅ 完全凍結 - 語言能力保留")
    else:
        print("   ⚠️  有參數可訓練（非 LoRA）")

    # LoRA adapters
    print(f"\n4. LoRA adapters:")
    print(f"   可訓練: {lora_trainable:,} ({lora_trainable/1e6:.2f}M)")
    if lora_trainable > 0:
        print("   ✅ 只訓練這些參數 - 不會破壞原始模型")
    else:
        print("   ❌ 沒有 LoRA 參數 - 設置失敗")

    print("\n" + "="*100)

    return projector_trainable == 0 and audio_trainable == 0 and lora_trainable > 0


# ============================================================================
# 第二部分：重用數據集（與 stage1_training_real_model.py 相同）
# ============================================================================

class Qwen2AudioTrainingDataset(Dataset):
    """
    使用實際 Qwen2-Audio 格式的訓練數據集
    """

    def __init__(self, data_list: List[Dict], processor):
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
                if audio_path.endswith('.npy'):
                    audio = np.load(audio_path).astype(np.float32)
                else:
                    audio, _ = librosa.load(
                        audio_path,
                        sr=self.processor.feature_extractor.sampling_rate
                    )
            except Exception as e:
                print(f"警告：無法加載音頻 {audio_path}: {e}")
                audio = np.random.randn(16000 * 3).astype(np.float32)
        else:
            audio = np.random.randn(16000 * 3).astype(np.float32)

        # 3. 處理輸入
        inputs = self.processor(
            text=text,
            audio=audio,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # 4. 創建 labels
        labels = inputs['input_ids'].clone()

        # 構建返回值
        result = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

        # 添加音頻特徵
        if 'input_features' in inputs:
            result['input_features'] = inputs['input_features'].squeeze(0)
        if 'feature_attention_mask' in inputs:
            result['feature_attention_mask'] = inputs['feature_attention_mask'].squeeze(0)

        return result


def load_dataset_from_disk(metadata_path, base_dir):
    """從磁盤加載數據集"""
    print(f"\n從磁盤加載數據集: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    base_path = Path(base_dir)

    data_list = []
    for item in metadata:
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
    """自定義 collate 函數"""
    max_text_len = max(item['input_ids'].shape[0] for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_text_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_text_len), dtype=torch.long)
    labels = torch.full((batch_size, max_text_len), -100, dtype=torch.long)

    has_audio = 'input_features' in batch[0]
    if has_audio:
        audio_shape = batch[0]['input_features'].shape
        input_features = torch.zeros((batch_size, *audio_shape), dtype=torch.float32)
        feature_attention_mask = torch.zeros((batch_size, audio_shape[1]), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i, :seq_len] = item['labels']

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
    """驗證函數"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
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


def train_with_lora(
    model,
    processor,
    train_loader,
    val_loader=None,
    num_epochs=2,
    learning_rate=1e-4,
    device='cuda'
):
    """
    使用 LoRA 進行訓練

    Args:
        model: 配置好 LoRA 的 Qwen2-Audio 模型
        processor: AutoProcessor
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        num_epochs: 訓練輪數
        learning_rate: 學習率
        device: 設備
    """

    print("\n" + "="*100)
    print("  開始 LoRA 訓練")
    print("="*100)

    # 移動模型到設備
    model = model.to(device)
    model.train()

    # 優化器（只優化 LoRA 參數）
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        print("❌ 錯誤：沒有可訓練的參數！")
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

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

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

        # 保存 checkpoint (只保存 LoRA adapters)
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f'checkpoint_lora_epoch_{epoch+1}'
            model.save_pretrained(checkpoint_path)
            print(f"✓ LoRA checkpoint 已保存: {checkpoint_path}\n")

    print("🎉 LoRA 訓練完成！\n")

    return model


# ============================================================================
# 第四部分：主程式
# ============================================================================

def main():
    """
    主函數 - 使用 LoRA 微調 Qwen2-Audio
    """

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║      Qwen2-Audio LoRA 微調 - 正確的微調方式                            ║
║                                                                          ║
║      與 stage1_training_real_model.py 的對比：                          ║
║                                                                          ║
║      ❌ stage1_training_real_model.py:                                  ║
║         - 訓練 multi_modal_projector (5.25M 參數)                       ║
║         - 破壞已有的音頻-文本對齊                                       ║
║         - 在隨機噪音上學習，導致性能下降                                ║
║                                                                          ║
║      ✅ 本腳本 (LoRA):                                                  ║
║         - 凍結 multi_modal_projector → 保留對齊                         ║
║         - 只在 LLM 添加 LoRA adapters (~3M 參數)                       ║
║         - 原始模型完全不變，可隨時恢復                                  ║
║         - 參數更少，訓練更快，不會遺忘                                  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    data_dir = './training_data'

    # ========================================================================
    # 步驟 1: 確定設備並加載模型
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 1: 加載 Qwen2-Audio 模型")
    print("="*100)

    # 檢查設備
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"\n✓ 使用 MPS GPU (Apple Silicon)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ 使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("\n⚠️  使用 CPU")

    print(f"\n正在加載模型...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    print("✓ 模型加載完成")

    # ========================================================================
    # 步驟 2: 設置 LoRA
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 2: 配置 LoRA")
    print("="*100)

    model = setup_lora_model(
        model,
        lora_rank=8,      # 可調整：4, 8, 16, 32
        lora_alpha=16,    # 通常 = 2 * rank
        target_modules=["q_proj", "v_proj"]  # 只在 Q 和 V 投影添加 LoRA
    )

    # 驗證設置
    if not verify_lora_setup(model):
        print("\n❌ LoRA 設置驗證失敗，請檢查配置")
        return

    # ========================================================================
    # 步驟 3: 準備數據
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 3: 準備訓練數據")
    print("="*100)

    train_metadata_path = Path(data_dir) / 'train' / 'metadata.json'
    val_metadata_path = Path(data_dir) / 'val' / 'metadata.json'

    if not train_metadata_path.exists():
        print(f"\n❌ 找不到訓練數據: {train_metadata_path}")
        print("請先運行 stage1_training_real_model.py 生成數據，或手動準備數據")
        return

    # 加載 processor
    print("\n加載 Processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 加載數據
    train_data = load_dataset_from_disk(train_metadata_path, Path(data_dir) / 'train')
    val_data = load_dataset_from_disk(val_metadata_path, Path(data_dir) / 'val')

    train_dataset = Qwen2AudioTrainingDataset(train_data, processor)
    val_dataset = Qwen2AudioTrainingDataset(val_data, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"\n✓ 訓練樣本: {len(train_dataset)}")
    print(f"✓ 驗證樣本: {len(val_dataset)}")

    # ========================================================================
    # 步驟 4: 開始訓練
    # ========================================================================
    print("\n" + "="*100)
    print("  步驟 4: 開始 LoRA 訓練")
    print("="*100)

    model = train_with_lora(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        learning_rate=1e-4,
        device=device
    )

    print("\n" + "="*100)
    print("  訓練完成")
    print("="*100)
    print("\n✅ LoRA 微調的優勢：")
    print("  1. multi_modal_projector 未被修改 → 音頻-文本對齊完整保留")
    print("  2. audio_tower 未被修改 → 音頻編碼能力不變")
    print("  3. language_model 原始權重不變 → 可隨時移除 LoRA 恢復原始模型")
    print("  4. 只訓練 ~3M 參數 → 比訓練 projector (5.25M) 更輕量")
    print("  5. 多個 LoRA adapters 可以共存 → 支持多任務")

    print("\n如何使用訓練好的模型：")
    print("  # 加載 LoRA adapters")
    print("  from peft import PeftModel")
    print("  base_model = Qwen2AudioForConditionalGeneration.from_pretrained(...)")
    print("  model = PeftModel.from_pretrained(base_model, 'checkpoint_lora_epoch_2')")
    print("\n  # 推理")
    print("  outputs = model.generate(...)")
    print("\n  # 移除 LoRA 恢復原始模型")
    print("  model = model.unload()")

    print("="*100 + "\n")


if __name__ == "__main__":
    main()
