"""
測試多人語音分離數據集

展示如何：
1. 讀取數據
2. 驗證數據正確性
3. 使用 PyTorch DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os


class MultiSpeakerDataset(Dataset):
    """多人語音分離數據集"""

    def __init__(self, data_dir, split='train', max_speakers=3):
        self.data_dir = os.path.join(data_dir, split)

        # 讀取 metadata
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.max_speakers = max_speakers
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
            'id': sample['id'],
            'mixed': torch.FloatTensor(mixed),
            'separated': torch.FloatTensor(np.stack(separated_list, axis=0)),
            'num_speakers': num_speakers,
            'speaker_mask': torch.FloatTensor(speaker_mask),
            'texts': [s['text'] for s in sample['speakers']]
        }


def test_basic_reading():
    """測試基本數據讀取"""
    print("\n" + "="*80)
    print("測試 1: 基本數據讀取")
    print("="*80)

    # 讀取 metadata
    with open('multi_speaker_data/train/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 測試第一個樣本
    sample = metadata[0]
    print(f"\n樣本 ID: {sample['id']}")
    print(f"說話者數量: {sample['num_speakers']}")
    print("說話者文本:")
    for speaker in sample['speakers']:
        print(f"  說話者 {speaker['speaker_id']}: {speaker['text']}")

    # 讀取混合特徵
    mixed = np.load('multi_speaker_data/train/' + sample['mixed_audio'])
    print(f"\n混合特徵形狀: {mixed.shape}")
    print(f"混合特徵範圍: [{mixed.min():.3f}, {mixed.max():.3f}]")

    # 讀取分離特徵
    for speaker in sample['speakers']:
        sep = np.load('multi_speaker_data/train/' + speaker['feature_file'])
        print(f"說話者 {speaker['speaker_id']} 特徵形狀: {sep.shape}")

    print("\n✅ 基本讀取測試通過")


def test_reconstruction():
    """測試混合特徵重建"""
    print("\n" + "="*80)
    print("測試 2: 混合特徵重建驗證")
    print("="*80)

    with open('multi_speaker_data/train/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 測試前 5 個樣本
    for i in range(min(5, len(metadata))):
        sample = metadata[i]

        # 讀取混合特徵
        mixed = np.load('multi_speaker_data/train/' + sample['mixed_audio'])

        # 讀取並重建
        separated_list = []
        for speaker in sample['speakers']:
            sep = np.load('multi_speaker_data/train/' + speaker['feature_file'])
            separated_list.append(sep)

        reconstructed = np.sum(separated_list, axis=0)

        # 驗證
        max_error = np.abs(mixed - reconstructed).max()
        is_valid = np.allclose(mixed, reconstructed)

        print(f"{sample['id']}: {sample['num_speakers']} 人, "
              f"最大誤差: {max_error:.10f}, "
              f"驗證: {'✓' if is_valid else '✗'}")

    print("\n✅ 重建驗證測試通過")


def test_dataset_class():
    """測試 Dataset 類"""
    print("\n" + "="*80)
    print("測試 3: PyTorch Dataset 類")
    print("="*80)

    dataset = MultiSpeakerDataset('multi_speaker_data', split='train')

    print(f"\n數據集大小: {len(dataset)}")

    # 測試第一個樣本
    sample = dataset[0]
    print(f"\n樣本 ID: {sample['id']}")
    print(f"Mixed shape: {sample['mixed'].shape}")
    print(f"Separated shape: {sample['separated'].shape}")
    print(f"Num speakers: {sample['num_speakers']}")
    print(f"Speaker mask: {sample['speaker_mask']}")
    print("Texts:")
    for i, text in enumerate(sample['texts']):
        print(f"  說話者 {i}: {text}")

    print("\n✅ Dataset 類測試通過")


def collate_fn(batch):
    """自定義 collate function 處理不同長度的 texts"""
    return {
        'id': [item['id'] for item in batch],
        'mixed': torch.stack([item['mixed'] for item in batch]),
        'separated': torch.stack([item['separated'] for item in batch]),
        'num_speakers': torch.LongTensor([item['num_speakers'] for item in batch]),
        'speaker_mask': torch.stack([item['speaker_mask'] for item in batch]),
        'texts': [item['texts'] for item in batch]  # 保持為列表
    }


def test_dataloader():
    """測試 DataLoader"""
    print("\n" + "="*80)
    print("測試 4: PyTorch DataLoader")
    print("="*80)

    train_dataset = MultiSpeakerDataset('multi_speaker_data', split='train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print(f"\nBatch 數量: {len(train_loader)}")

    # 測試第一個 batch
    batch = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"  Mixed: {batch['mixed'].shape}")          # [4, 100, 1280]
    print(f"  Separated: {batch['separated'].shape}")  # [4, 3, 100, 1280]
    print(f"  Num speakers: {batch['num_speakers']}")
    print(f"  Speaker mask: {batch['speaker_mask'].shape}")

    print(f"\nBatch IDs: {batch['id']}")
    print("\n第一個樣本的說話者文本:")
    for text in batch['texts'][0]:
        print(f"  - {text}")

    print("\n✅ DataLoader 測試通過")


def test_statistics():
    """統計數據集信息"""
    print("\n" + "="*80)
    print("測試 5: 數據集統計")
    print("="*80)

    for split in ['train', 'val']:
        with open(f'multi_speaker_data/{split}/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        total_samples = len(metadata)
        total_speakers = sum(m['num_speakers'] for m in metadata)
        avg_speakers = total_speakers / total_samples

        # 統計說話者數量分佈
        speaker_counts = {}
        for sample in metadata:
            n = sample['num_speakers']
            speaker_counts[n] = speaker_counts.get(n, 0) + 1

        print(f"\n{split.upper()} 集:")
        print(f"  總樣本數: {total_samples}")
        print(f"  總說話者數: {total_speakers}")
        print(f"  平均說話者數: {avg_speakers:.2f}")
        print(f"  說話者分佈:")
        for n in sorted(speaker_counts.keys()):
            print(f"    {n} 人: {speaker_counts[n]} 個樣本 ({speaker_counts[n]/total_samples*100:.1f}%)")

    print("\n✅ 統計測試通過")


def main():
    """運行所有測試"""
    print("\n" + "="*80)
    print("  多人語音分離數據集測試")
    print("="*80)

    try:
        test_basic_reading()
        test_reconstruction()
        test_dataset_class()
        test_dataloader()
        test_statistics()

        print("\n" + "="*80)
        print("  ✅ 所有測試通過！")
        print("="*80)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
