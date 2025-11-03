"""
測試空間音頻定位模型

載入訓練好的模型，在測試集上評估
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import re

from spatial_audio_model import Qwen2AudioWithSpatialToken
from train_spatial_audio import SpatialAudioDataset


def extract_angle_from_text(text):
    """
    從生成的文字中提取角度

    例如:
    - "來自右側大約 60 度" → 60.0
    - "位於左側 45 度" → -45.0
    """
    # 移除多餘空格
    text = ' '.join(text.split())

    # 查找角度數字
    angle_match = re.search(r'(\d+(?:\.\d+)?)\s*度', text)

    if not angle_match:
        return None

    angle = float(angle_match.group(1))

    # 判斷左右
    if '左' in text or '左側' in text or '左方' in text:
        angle = -angle
    elif '右' in text or '右側' in text or '右方' in text:
        angle = angle
    elif '正前方' in text:
        angle = 0.0
    else:
        # 無法判斷方向
        return None

    return angle


def evaluate_model(model, dataset, device, num_samples=None):
    """
    評估模型

    Args:
        model: 訓練好的模型
        dataset: 測試數據集
        device: 設備
        num_samples: 測試樣本數（None = 全部）

    Returns:
        results: 評估結果
    """
    model.eval()

    if num_samples is None:
        num_samples = len(dataset)

    results = {
        'predictions': [],
        'ground_truths': [],
        'errors': [],
        'samples': []
    }

    print(f"\n評估 {num_samples} 個樣本...")

    for i in tqdm(range(num_samples)):
        sample = dataset[i]

        # 生成預測
        with torch.no_grad():
            response = model.generate(
                left_audio=sample['left_audio'],
                right_audio=sample['right_audio'],
                text_prompt=sample['prompt'],
                max_new_tokens=128
            )

        # 提取預測角度
        pred_angle = extract_angle_from_text(response)
        gt_angle = sample['angle']

        # 計算誤差
        if pred_angle is not None:
            error = abs(pred_angle - gt_angle)
        else:
            error = None  # 無法提取角度

        # 記錄
        results['predictions'].append(pred_angle)
        results['ground_truths'].append(gt_angle)
        results['errors'].append(error)

        results['samples'].append({
            'id': sample['id'],
            'ground_truth': gt_angle,
            'prediction': pred_angle,
            'error': error,
            'prompt': sample['prompt'],
            'response': response,
            'expected': sample['response']
        })

    return results


def compute_metrics(results):
    """計算評估指標"""
    # 過濾掉無法提取角度的樣本
    valid_errors = [e for e in results['errors'] if e is not None]

    if len(valid_errors) == 0:
        print("警告：沒有有效的預測角度！")
        return {}

    valid_errors = np.array(valid_errors)

    metrics = {
        'num_valid': len(valid_errors),
        'num_total': len(results['errors']),
        'extraction_rate': len(valid_errors) / len(results['errors']) * 100,
        'mae': float(valid_errors.mean()),
        'rmse': float(np.sqrt((valid_errors ** 2).mean())),
        'median_error': float(np.median(valid_errors)),
        'std': float(valid_errors.std()),
        'accuracy_5': float((valid_errors < 5).mean() * 100),
        'accuracy_10': float((valid_errors < 10).mean() * 100),
        'accuracy_15': float((valid_errors < 15).mean() * 100),
        'accuracy_20': float((valid_errors < 20).mean() * 100),
    }

    return metrics


def print_metrics(metrics):
    """打印評估指標"""
    print("\n" + "=" * 80)
    print("  評估結果")
    print("=" * 80)

    print(f"\n樣本統計:")
    print(f"  總樣本數: {metrics['num_total']}")
    print(f"  有效預測: {metrics['num_valid']}")
    print(f"  提取率: {metrics['extraction_rate']:.1f}%")

    print(f"\n角度預測誤差:")
    print(f"  MAE (平均絕對誤差): {metrics['mae']:.2f}°")
    print(f"  RMSE (均方根誤差): {metrics['rmse']:.2f}°")
    print(f"  Median Error (中位數誤差): {metrics['median_error']:.2f}°")
    print(f"  Std (標準差): {metrics['std']:.2f}°")

    print(f"\n準確率 (容忍度):")
    print(f"  Accuracy@5°:  {metrics['accuracy_5']:.1f}%")
    print(f"  Accuracy@10°: {metrics['accuracy_10']:.1f}%")
    print(f"  Accuracy@15°: {metrics['accuracy_15']:.1f}%")
    print(f"  Accuracy@20°: {metrics['accuracy_20']:.1f}%")


def print_sample_predictions(results, num_samples=5):
    """打印樣本預測"""
    print("\n" + "=" * 80)
    print(f"  樣本預測 (前 {num_samples} 個)")
    print("=" * 80)

    for i, sample in enumerate(results['samples'][:num_samples]):
        print(f"\n樣本 {i+1}: {sample['id']}")
        print(f"  Ground Truth: {sample['ground_truth']:.1f}°")
        print(f"  Prediction: {sample['prediction']:.1f}°" if sample['prediction'] is not None else "  Prediction: N/A")
        print(f"  Error: {sample['error']:.1f}°" if sample['error'] is not None else "  Error: N/A")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Expected: {sample['expected']}")
        print(f"  Response: {sample['response']}")


def save_results(results, metrics, output_path):
    """保存評估結果"""
    output = {
        'metrics': metrics,
        'samples': results['samples']
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 結果已保存: {output_path}")


def test(args):
    """主測試函數"""
    print("=" * 80)
    print("  測試空間音頻定位模型")
    print("=" * 80)

    # 設備
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n使用設備: {device}")

    # ═══════════════════════════════════════
    # 載入模型
    # ═══════════════════════════════════════

    print(f"\n載入模型...")
    print(f"  Checkpoint: {args.checkpoint}")

    # 載入 checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 獲取訓練配置
    train_args = checkpoint.get('args', {})

    # 創建模型
    model = Qwen2AudioWithSpatialToken(
        pretrained_model_name=train_args.get('model_name', 'Qwen/Qwen2-Audio-7B-Instruct'),
        spatial_dim=train_args.get('spatial_dim', 256),
        use_lora=train_args.get('use_lora', False),
        lora_r=train_args.get('lora_r', 16),
        lora_alpha=train_args.get('lora_alpha', 32)
    )

    # 載入權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ 模型已載入 (Epoch {checkpoint['epoch']})")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    # ═══════════════════════════════════════
    # 載入測試數據
    # ═══════════════════════════════════════

    print(f"\n載入測試數據...")

    test_dataset = SpatialAudioDataset(args.data_dir, split=args.split)

    # ═══════════════════════════════════════
    # 評估
    # ═══════════════════════════════════════

    results = evaluate_model(
        model,
        test_dataset,
        device,
        num_samples=args.num_samples
    )

    # ═══════════════════════════════════════
    # 計算指標
    # ═══════════════════════════════════════

    metrics = compute_metrics(results)

    # 打印結果
    print_metrics(metrics)
    print_sample_predictions(results, num_samples=args.show_samples)

    # ═══════════════════════════════════════
    # 保存結果
    # ═══════════════════════════════════════

    if args.output:
        save_results(results, metrics, args.output)

    print("\n" + "=" * 80)
    print("  測試完成！")
    print("=" * 80)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='測試空間音頻定位模型')

    # 模型
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路徑')

    # 數據
    parser.add_argument('--data_dir', type=str, default='spatial_audio_data',
                        help='數據目錄')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='測試集 (train 或 val)')

    # 測試設定
    parser.add_argument('--num_samples', type=int, default=None,
                        help='測試樣本數 (None = 全部)')
    parser.add_argument('--show_samples', type=int, default=5,
                        help='顯示多少個樣本預測')

    # 輸出
    parser.add_argument('--output', type=str, default='test_results.json',
                        help='輸出結果檔案')

    args = parser.parse_args()

    # 測試
    test(args)


if __name__ == '__main__':
    main()
