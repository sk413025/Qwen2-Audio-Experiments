"""
生成多人語音分離合成數據集

資料夾結構：
multi_speaker_data/
├── train/
│   ├── metadata.json
│   ├── mixed/          # 混合音頻特徵
│   └── separated/      # 分離的音頻特徵
└── val/
    ├── metadata.json
    ├── mixed/
    └── separated/

數據格式：
- 混合音頻：.npy 格式，可以直接用 numpy 讀取檢查
- 分離音頻：.npy 格式，每個說話者一個文件
- metadata.json：包含對話格式、說話者信息等
"""

import os
import json
import numpy as np
from typing import List, Dict


def generate_synthetic_audio_features(
    seq_len: int = 100,
    feature_dim: int = 1280,
    speaker_id: int = 0,
    seed: int = 42
) -> np.ndarray:
    """
    生成合成的音頻特徵（模擬 Whisper 特徵）

    使用獨特的正弦波模式來模擬不同說話者的特徵

    Args:
        seq_len: 序列長度
        feature_dim: 特徵維度
        speaker_id: 說話者 ID（用於生成獨特模式）
        seed: 隨機種子

    Returns:
        特徵數組 [seq_len, feature_dim]
    """
    np.random.seed(seed)

    # 為每個說話者創建獨特的頻率模式
    base_freq = (speaker_id + 1) * 0.1
    t = np.linspace(0, 10, seq_len)
    feature = np.zeros((seq_len, feature_dim))

    for dim in range(feature_dim):
        freq = base_freq + (dim % 10) * 0.01
        phase = np.random.rand() * 2 * np.pi

        # 組合多個正弦波
        feature[:, dim] = (
            np.sin(2 * np.pi * freq * t + phase) +
            0.3 * np.sin(4 * np.pi * freq * t + phase) +
            0.1 * np.random.randn(seq_len)
        )

    # 標準化
    feature = feature / (np.std(feature) + 1e-8)

    return feature


# 定義多個說話者的句子
SPEAKER_SENTENCES = {
    'speaker_1': [
        "公司接到一份國外訂單。",
        "今天會議討論了新的銷售策略。",
        "我們需要在月底前完成這個項目。",
        "市場部提出了創新的行銷方案。",
        "財務報表顯示本季度業績良好。",
    ],
    'speaker_2': [
        "他在禮堂主持開幕典禮。",
        "學校邀請專家來演講。",
        "圖書館新增了許多藏書。",
        "體育館正在進行裝修工程。",
        "校長宣布下週舉辦運動會。",
    ],
    'speaker_3': [
        "這學期學校有書法比賽。",
        "美術課要準備期末作品展。",
        "音樂社團正在排練新曲目。",
        "科學實驗課非常有趣。",
        "同學們積極參與社團活動。",
    ]
}


def generate_multi_speaker_sample(
    sample_id: str,
    num_speakers: int,
    seq_len: int = 100,
    feature_dim: int = 1280,
    seed: int = 42
) -> Dict:
    """
    生成一個多人語音樣本

    Returns:
        包含混合特徵、分離特徵和元數據的字典
    """
    np.random.seed(seed)

    # 隨機選擇說話者（2-num_speakers 人）
    actual_num_speakers = np.random.randint(2, num_speakers + 1)
    speaker_ids = list(range(actual_num_speakers))

    # 生成每個說話者的特徵和文本
    separated_features = []
    speakers_info = []

    for i, speaker_id in enumerate(speaker_ids):
        # 生成特徵
        feature = generate_synthetic_audio_features(
            seq_len=seq_len,
            feature_dim=feature_dim,
            speaker_id=speaker_id,
            seed=seed + i
        )
        separated_features.append(feature)

        # 隨機選擇一句話
        speaker_key = f'speaker_{speaker_id + 1}'
        sentence_idx = np.random.randint(0, len(SPEAKER_SENTENCES[speaker_key]))
        text = SPEAKER_SENTENCES[speaker_key][sentence_idx]

        speakers_info.append({
            'speaker_id': i,
            'text': text,
            'feature_file': f'separated/{sample_id}_speaker_{i}.npy'
        })

    # 混合特徵
    mixed_features = np.sum(separated_features, axis=0)

    # 構建元數據
    metadata = {
        'id': sample_id,
        'mixed_audio': f'mixed/{sample_id}.npy',
        'num_speakers': actual_num_speakers,
        'speakers': speakers_info,
        'conversation': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'audio',
                        'audio_url': f'{sample_id}.npy'
                    },
                    {
                        'type': 'text',
                        'text': '這段音頻包含多人同時說話，請分離出每個人的內容。'
                    }
                ]
            },
            {
                'role': 'assistant',
                'content': f"這段音頻包含 {actual_num_speakers} 個說話者：\n" +
                          "\n".join([f"說話者 {i+1}：{s['text']}" for i, s in enumerate(speakers_info)])
            }
        ],
        'seq_len': seq_len,
        'feature_dim': feature_dim
    }

    return {
        'metadata': metadata,
        'mixed_features': mixed_features,
        'separated_features': separated_features
    }


def create_dataset(
    output_dir: str,
    split: str,
    num_samples: int,
    max_speakers: int = 3,
    seq_len: int = 100,
    feature_dim: int = 1280,
    seed: int = 42
):
    """
    創建數據集（train 或 val）

    Args:
        output_dir: 輸出目錄
        split: 'train' 或 'val'
        num_samples: 樣本數量
        max_speakers: 最大說話者數量
        seq_len: 序列長度
        feature_dim: 特徵維度
        seed: 隨機種子
    """
    print(f"\n生成 {split} 數據集...")
    print(f"  樣本數: {num_samples}")
    print(f"  最大說話者數: {max_speakers}")

    # 創建目錄
    split_dir = os.path.join(output_dir, split)
    mixed_dir = os.path.join(split_dir, 'mixed')
    separated_dir = os.path.join(split_dir, 'separated')

    os.makedirs(mixed_dir, exist_ok=True)
    os.makedirs(separated_dir, exist_ok=True)

    # 生成樣本
    metadata_list = []

    for i in range(num_samples):
        sample_id = f'{split}_{i:04d}'

        # 生成樣本
        sample = generate_multi_speaker_sample(
            sample_id=sample_id,
            num_speakers=max_speakers,
            seq_len=seq_len,
            feature_dim=feature_dim,
            seed=seed + i
        )

        # 保存混合特徵
        mixed_path = os.path.join(mixed_dir, f'{sample_id}.npy')
        np.save(mixed_path, sample['mixed_features'])

        # 保存分離特徵
        for speaker_idx, speaker_feature in enumerate(sample['separated_features']):
            separated_path = os.path.join(separated_dir, f'{sample_id}_speaker_{speaker_idx}.npy')
            np.save(separated_path, speaker_feature)

        # 添加到 metadata
        metadata_list.append(sample['metadata'])

        if (i + 1) % 10 == 0:
            print(f"  已生成 {i+1}/{num_samples}")

    # 保存 metadata.json
    metadata_path = os.path.join(split_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print(f"✓ {split} 數據集生成完成")
    print(f"  混合音頻: {len(metadata_list)} 個")
    print(f"  總分離音頻: {sum(len(m['speakers']) for m in metadata_list)} 個")
    print(f"  metadata 保存到: {metadata_path}")


def main():
    """主函數"""
    print("="*80)
    print("  生成多人語音分離合成數據集")
    print("="*80)

    output_dir = 'multi_speaker_data'

    # 生成訓練集
    create_dataset(
        output_dir=output_dir,
        split='train',
        num_samples=30,
        max_speakers=3,
        seq_len=100,
        feature_dim=1280,
        seed=42
    )

    # 生成驗證集
    create_dataset(
        output_dir=output_dir,
        split='val',
        num_samples=10,
        max_speakers=3,
        seq_len=100,
        feature_dim=1280,
        seed=999
    )

    print("\n" + "="*80)
    print("  ✅ 數據集生成完成！")
    print("="*80)
    print(f"\n資料夾結構：")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── metadata.json")
    print(f"  │   ├── mixed/         (混合音頻特徵 .npy)")
    print(f"  │   └── separated/     (分離音頻特徵 .npy)")
    print(f"  └── val/")
    print(f"      ├── metadata.json")
    print(f"      ├── mixed/")
    print(f"      └── separated/")

    # 顯示統計
    print("\n數據集統計：")
    for split in ['train', 'val']:
        metadata_path = os.path.join(output_dir, split, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        total_speakers = sum(m['num_speakers'] for m in metadata)
        avg_speakers = total_speakers / len(metadata)

        print(f"  {split}:")
        print(f"    樣本數: {len(metadata)}")
        print(f"    總說話者數: {total_speakers}")
        print(f"    平均說話者數: {avg_speakers:.2f}")

    # 顯示檢查方法
    print("\n" + "="*80)
    print("  如何檢查數據：")
    print("="*80)
    print("""
  1. 檢查 metadata.json：
     cat multi_speaker_data/train/metadata.json | head -50

  2. 用 Python 讀取音頻特徵：
     import numpy as np
     mixed = np.load('multi_speaker_data/train/mixed/train_0000.npy')
     print(f"混合特徵形狀: {mixed.shape}")  # [100, 1280]

     speaker_0 = np.load('multi_speaker_data/train/separated/train_0000_speaker_0.npy')
     print(f"說話者 0 特徵形狀: {speaker_0.shape}")  # [100, 1280]

  3. 驗證混合關係：
     # 混合特徵應該等於所有說話者特徵的和
     reconstructed = speaker_0 + speaker_1 + ...
     assert np.allclose(mixed, reconstructed)
    """)


if __name__ == '__main__':
    main()
