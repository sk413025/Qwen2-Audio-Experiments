"""
訓練空間音頻定位模型

兩階段訓練:
1. 訓練 Spatial Extractor + Projector (凍結 Qwen2-Audio)
2. LoRA 微調 LLM (可選)
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from spatial_audio_model import Qwen2AudioWithSpatialToken


class SpatialAudioDataset(Dataset):
    """
    空間音頻數據集

    從 spatial_audio_data 載入
    """
    def __init__(self, data_dir='spatial_audio_data', split='train'):
        self.data_dir = Path(data_dir) / split
        self.split = split

        # 讀取 metadata
        metadata_path = self.data_dir / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"✓ 載入 {split} 數據集: {len(self.metadata)} 個樣本")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata[idx]

        # 載入音頻
        left_audio = np.load(self.data_dir / sample['left_audio'])
        right_audio = np.load(self.data_dir / sample['right_audio'])

        # 獲取對話
        conversation = sample['conversation']
        prompt = conversation[0]['content']
        response = conversation[1]['content']

        return {
            'id': sample['id'],
            'left_audio': left_audio.astype(np.float32),
            'right_audio': right_audio.astype(np.float32),
            'prompt': prompt,
            'response': response,
            'angle': sample['angle']
        }


def collate_fn(batch, processor, spatial_token_id, device='cpu'):
    """
    自定義 collate function

    只返回原始數據，讓 model.forward() 處理
    """
    left_audios = [item['left_audio'] for item in batch]
    right_audios = [item['right_audio'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]

    return {
        'left_audios': left_audios,
        'right_audios': right_audios,
        'prompts': prompts,
        'responses': responses
    }


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch in progress_bar:
        # 前向傳播
        outputs = model(
            left_audio=batch['left_audios'],
            right_audio=batch['right_audios'],
            text_prompt=batch['prompts'],
            text_response=batch['responses']  # 訓練模式需要 responses
        )

        loss = outputs.loss

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 統計
        total_loss += loss.item()
        num_batches += 1

        # 更新進度條
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, device, epoch):
    """驗證"""
    model.eval()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(val_loader, desc=f'Validation {epoch}')

    for batch in progress_bar:
        try:
            outputs = model(
                left_audio=batch['left_audios'],
                right_audio=batch['right_audios'],
                text_prompt=batch['prompts'],
                text_response=batch['responses']  # 驗證模式也需要 responses 計算 loss
            )

            loss = outputs.loss

            # 檢查 loss 是否為 nan 或 inf
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'loss': loss.item()})
            else:
                progress_bar.set_postfix({'loss': 'nan/inf'})
        except Exception as e:
            print(f"\n警告：驗證批次跳過，錯誤：{e}")
            continue

    if num_batches == 0:
        return float('nan')

    avg_loss = total_loss / num_batches
    return avg_loss


def plot_training_history(history, save_path='training_history.png'):
    """繪製訓練曲線"""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✓ 訓練曲線已保存: {save_path}")


def train(args):
    """主訓練函數"""
    print("=" * 80)
    print("  訓練空間音頻定位模型")
    print("=" * 80)

    # 設備
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n使用設備: {device}")

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════
    # 載入數據
    # ═══════════════════════════════════════

    print(f"\n載入數據集...")

    train_dataset = SpatialAudioDataset(args.data_dir, split='train')
    val_dataset = SpatialAudioDataset(args.data_dir, split='val')

    # ═══════════════════════════════════════
    # 創建模型
    # ═══════════════════════════════════════

    print(f"\n創建模型...")

    model = Qwen2AudioWithSpatialToken(
        pretrained_model_name=args.model_name,
        spatial_dim=args.spatial_dim,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    model = model.to(device)

    # 統計參數
    params_info = model.get_trainable_params()
    print(f"\n可訓練參數:")
    print(f"  Spatial Extractor: {params_info['spatial_extractor']:,}")
    print(f"  Spatial Projector: {params_info['spatial_projector']:,}")
    if params_info['lora'] > 0:
        print(f"  LoRA: {params_info['lora']:,}")
    print(f"  總計: {params_info['total']:,}")

    # ═══════════════════════════════════════
    # 準備訓練
    # ═══════════════════════════════════════

    # 優化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 學習率調度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.1
    )

    # DataLoader
    def collate_wrapper(batch):
        return collate_fn(
            batch,
            model.processor,
            model.spatial_token_id,
            device
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_wrapper,
        num_workers=0  # MPS 不支援多進程
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=0
    )

    # ═══════════════════════════════════════
    # 訓練循環
    # ═══════════════════════════════════════

    print(f"\n開始訓練...")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*80}")

        # 訓練
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # 驗證
        val_loss = validate(model, val_loader, device, epoch)

        # 更新學習率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 記錄
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        print(f"\nEpoch {epoch} 結果:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # 準備 checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history,
            'args': vars(args)
        }

        # 保存最佳模型
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ 最佳模型已保存 (Val Loss: {val_loss:.4f})")

        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint 已保存: {checkpoint_path}")

    # ═══════════════════════════════════════
    # 訓練完成
    # ═══════════════════════════════════════

    print(f"\n{'='*80}")
    print("  訓練完成！")
    print(f"{'='*80}")
    print(f"\n最佳驗證 Loss: {best_val_loss:.4f}")

    # 保存訓練歷史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"✓ 訓練歷史已保存: {history_path}")

    # 繪製訓練曲線
    plot_training_history(history, output_dir / 'training_history.png')

    return model, history


def main():
    parser = argparse.ArgumentParser(description='訓練空間音頻定位模型')

    # 數據
    parser.add_argument('--data_dir', type=str, default='spatial_audio_data',
                        help='數據目錄')

    # 模型
    parser.add_argument('--model_name', type=str,
                        default='Qwen/Qwen2-Audio-7B-Instruct',
                        help='預訓練模型名稱')
    parser.add_argument('--spatial_dim', type=int, default=256,
                        help='空間特徵維度')

    # LoRA
    parser.add_argument('--use_lora', action='store_true',
                        help='是否使用 LoRA 微調')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')

    # 訓練
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='訓練 epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # 輸出
    parser.add_argument('--output_dir', type=str, default='checkpoints/spatial_audio',
                        help='輸出目錄')
    parser.add_argument('--save_every', type=int, default=5,
                        help='每隔多少 epoch 保存一次')

    args = parser.parse_args()

    # 訓練
    train(args)


if __name__ == '__main__':
    main()
