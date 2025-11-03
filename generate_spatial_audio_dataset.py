"""
生成空間音頻定位數據集

使用 HRTF 模擬從單聲道音頻生成雙耳音頻
數據格式與 multi_speaker_data 一致
"""

import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import librosa
from scipy.signal import butter, lfilter
import random


def apply_itd(audio, itd_samples, sr=16000):
    """
    應用 ITD (Interaural Time Difference)

    Args:
        audio: [samples] 音頻信號
        itd_samples: ITD (樣本數，可以是小數)
        sr: 採樣率

    Returns:
        delayed_audio: 延遲後的音頻
    """
    if abs(itd_samples) < 0.01:
        return audio

    # 整數延遲
    int_delay = int(round(itd_samples))

    if int_delay > 0:
        # 正延遲：在前面補零
        delayed = np.pad(audio, (int_delay, 0), mode='constant')[:-int_delay]
    elif int_delay < 0:
        # 負延遲：在後面補零
        delayed = np.pad(audio, (0, -int_delay), mode='constant')[-int_delay:]
    else:
        delayed = audio

    return delayed


def apply_ild_filter(audio, ild_db, sr=16000):
    """
    應用 ILD (Interaural Level Difference)

    ILD 在高頻更明顯（頭部遮擋效應）

    Args:
        audio: [samples] 音頻信號
        ild_db: ILD (dB)，正值表示增益，負值表示衰減
        sr: 採樣率

    Returns:
        filtered_audio: 應用 ILD 後的音頻
    """
    if abs(ild_db) < 0.1:
        return audio

    # 應用增益
    linear_gain = 10 ** (ild_db / 20)
    audio_gained = audio * linear_gain

    # 如果是遠端耳朵（負 ILD），應用低通濾波（模擬高頻衰減）
    if ild_db < 0:
        # 截止頻率隨 ILD 變化
        cutoff_freq = 8000 + ild_db * 200  # 越負，截止頻率越低
        cutoff_freq = np.clip(cutoff_freq, 2000, 8000)

        # 設計低通濾波器
        nyquist = sr / 2
        normalized_cutoff = cutoff_freq / nyquist

        if normalized_cutoff < 1.0:
            b, a = butter(2, normalized_cutoff, btype='low')
            audio_gained = lfilter(b, a, audio_gained)

    return audio_gained


def generate_binaural_audio(mono_audio, angle_deg, sr=16000):
    """
    使用簡化 HRTF 模型生成雙耳音頻

    Args:
        mono_audio: [samples] 單聲道音頻
        angle_deg: 方位角（度），-90 (左) 到 +90 (右)
        sr: 採樣率

    Returns:
        left_audio: [samples] 左耳音頻
        right_audio: [samples] 右耳音頻
    """
    angle_rad = np.deg2rad(angle_deg)

    # === ITD 計算 (Woodworth 公式) ===
    head_radius = 0.0875  # m
    sound_speed = 343.0   # m/s

    itd_seconds = (head_radius / sound_speed) * (np.sin(angle_rad) + angle_rad)
    itd_samples = itd_seconds * sr

    # === ILD 計算 (簡化模型) ===
    # ILD 與 sin(θ) 成正比，最大約 ±15 dB
    ild_max = 15  # dB
    ild_db = ild_max * np.sin(angle_rad)

    # === 應用到音頻 ===
    if angle_deg >= 0:
        # 聲源在右側
        # 右耳：近端，增益，無延遲
        # 左耳：遠端，衰減，延遲
        right_audio = apply_ild_filter(mono_audio, ild_db / 2, sr)
        right_audio = apply_itd(right_audio, -itd_samples / 2, sr)

        left_audio = apply_ild_filter(mono_audio, -ild_db / 2, sr)
        left_audio = apply_itd(left_audio, itd_samples / 2, sr)
    else:
        # 聲源在左側
        left_audio = apply_ild_filter(mono_audio, -ild_db / 2, sr)  # 負的負 = 正增益
        left_audio = apply_itd(left_audio, -itd_samples / 2, sr)

        right_audio = apply_ild_filter(mono_audio, ild_db / 2, sr)  # 負 = 衰減
        right_audio = apply_itd(right_audio, itd_samples / 2, sr)

    # 確保長度一致
    min_len = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_len]
    right_audio = right_audio[:min_len]

    return left_audio, right_audio


def get_direction_description(angle):
    """
    將角度轉換為自然語言描述

    Args:
        angle: -90 to 90 度

    Returns:
        description: 方向描述
    """
    abs_angle = abs(angle)

    if angle > 5:
        direction = "右側"
    elif angle < -5:
        direction = "左側"
    else:
        return "正前方"

    # 角度描述
    if abs_angle < 15:
        return f"{direction}偏前方大約 {abs_angle:.0f} 度"
    elif abs_angle < 45:
        return f"{direction}前方大約 {abs_angle:.0f} 度"
    elif abs_angle < 75:
        return f"{direction}大約 {abs_angle:.0f} 度"
    else:
        return f"{direction}偏側方大約 {abs_angle:.0f} 度"


def generate_synthetic_audio(duration=3.0, sr=16000, audio_type='speech'):
    """
    生成合成音頻（用於測試）

    實際使用時應該載入真實音頻

    Args:
        duration: 時長（秒）
        sr: 採樣率
        audio_type: 音頻類型

    Returns:
        audio: [samples] 音頻信號
        description: 音頻內容描述
    """
    n_samples = int(duration * sr)

    if audio_type == 'speech':
        # 模擬語音：多個頻率的正弦波
        t = np.linspace(0, duration, n_samples)
        audio = 0
        for freq in [200, 400, 600, 800]:
            audio += 0.2 * np.sin(2 * np.pi * freq * t)

        # 添加包絡
        envelope = np.exp(-3 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio *= envelope

        descriptions = ["男性說話聲", "女性說話聲", "對話聲"]
        description = random.choice(descriptions)

    elif audio_type == 'music':
        # 模擬音樂
        t = np.linspace(0, duration, n_samples)
        audio = 0
        for freq in [440, 554, 659]:  # A, C#, E (A major chord)
            audio += 0.3 * np.sin(2 * np.pi * freq * t)

        descriptions = ["鋼琴演奏", "吉他彈奏", "音樂播放"]
        description = random.choice(descriptions)

    elif audio_type == 'environmental':
        # 模擬環境音
        # 使用白噪音 + 濾波
        audio = np.random.randn(n_samples) * 0.1

        # 低通濾波
        b, a = butter(4, 0.2)
        audio = lfilter(b, a, audio)

        descriptions = ["風聲", "水流聲", "鳥叫聲", "雨聲"]
        description = random.choice(descriptions)

    else:
        raise ValueError(f"Unknown audio type: {audio_type}")

    # 歸一化
    audio = audio / (np.abs(audio).max() + 1e-8)
    audio = audio * 0.8  # 避免削波

    return audio, description


def create_dataset(
    output_dir='spatial_audio_data',
    split='train',
    num_samples=30,
    duration=3.0,
    sr=16000,
    seed=42
):
    """
    創建數據集（train 或 val）

    Args:
        output_dir: 輸出目錄
        split: 'train' or 'val'
        num_samples: 樣本數量
        duration: 音頻時長（秒）
        sr: 採樣率
        seed: 隨機種子
    """
    np.random.seed(seed)
    random.seed(seed)

    # 創建目錄
    split_dir = Path(output_dir) / split
    left_dir = split_dir / 'left'
    right_dir = split_dir / 'right'
    mono_dir = split_dir / 'mono'

    for d in [left_dir, right_dir, mono_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metadata = []

    # 音頻類型分佈
    audio_types = ['speech'] * 12 + ['music'] * 9 + ['environmental'] * 9

    print(f"\n生成 {split} 集: {num_samples} 個樣本")

    for i in tqdm(range(num_samples)):
        sample_id = f"{split}_{i:04d}"

        # 隨機角度（-90 到 +90）
        angle = np.random.uniform(-90, 90)

        # 隨機選擇音頻類型
        audio_type = random.choice(audio_types)

        # 生成單聲道音頻
        mono_audio, audio_description = generate_synthetic_audio(
            duration=duration,
            sr=sr,
            audio_type=audio_type
        )

        # 生成雙耳音頻
        left_audio, right_audio = generate_binaural_audio(mono_audio, angle, sr)

        # 保存音頻
        np.save(left_dir / f"{sample_id}.npy", left_audio.astype(np.float32))
        np.save(right_dir / f"{sample_id}.npy", right_audio.astype(np.float32))
        np.save(mono_dir / f"{sample_id}.npy", mono_audio.astype(np.float32))

        # 方向描述
        direction_desc = get_direction_description(angle)

        # 構建對話
        # 多種提示格式
        prompts = [
            "請描述這段音頻的內容以及聲源的方向位置。",
            "這個聲音是什麼？它來自哪個方向？",
            "分析這段音頻，包括內容和空間位置。",
            "請識別音頻內容並判斷聲源方向。"
        ]
        prompt = random.choice(prompts)

        # 多種回答格式
        responses = [
            f"這是{audio_description}，來自{direction_desc}。",
            f"音頻內容是{audio_description}，聲源位於{direction_desc}。",
            f"我聽到{audio_description}，方向在{direction_desc}。",
            f"這段音頻是{audio_description}。從左右聲道的差異來看，聲源位於{direction_desc}。"
        ]
        response = random.choice(responses)

        # 元數據
        sample_meta = {
            'id': sample_id,
            'left_audio': f'left/{sample_id}.npy',
            'right_audio': f'right/{sample_id}.npy',
            'mono_audio': f'mono/{sample_id}.npy',
            'angle': float(angle),
            'audio_type': audio_type,
            'audio_description': audio_description,
            'direction_description': direction_desc,
            'conversation': [
                {
                    'role': 'user',
                    'content': prompt
                },
                {
                    'role': 'assistant',
                    'content': response
                }
            ],
            'duration': duration,
            'sample_rate': sr
        }

        metadata.append(sample_meta)

    # 保存 metadata
    metadata_path = split_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ {split} 集生成完成")
    print(f"  - 樣本數: {num_samples}")
    print(f"  - 保存位置: {split_dir}")

    return metadata


def generate_statistics(metadata):
    """生成數據集統計信息"""
    angles = [s['angle'] for s in metadata]

    stats = {
        'total_samples': len(metadata),
        'angle_range': [min(angles), max(angles)],
        'angle_mean': np.mean(angles),
        'angle_std': np.std(angles),
        'audio_type_distribution': {}
    }

    # 統計音頻類型
    for sample in metadata:
        audio_type = sample['audio_type']
        stats['audio_type_distribution'][audio_type] = \
            stats['audio_type_distribution'].get(audio_type, 0) + 1

    return stats


def create_readme(output_dir, train_stats, val_stats):
    """創建 README.md"""
    readme_content = f"""# 空間音頻定位數據集
## 使用 HRTF 模擬生成的雙耳音頻

**生成時間**: {np.datetime64('now')}
**任務**: 從雙聲道音頻預測聲源方向

---

## 📁 資料夾結構

```
spatial_audio_data/
├── README.md           (本文件)
├── train/
│   ├── metadata.json   (訓練集元數據)
│   ├── left/           (左聲道音頻 .npy)
│   ├── right/          (右聲道音頻 .npy)
│   └── mono/           (混合音頻 .npy)
└── val/
    ├── metadata.json   (驗證集元數據)
    ├── left/           (左聲道音頻 .npy)
    ├── right/          (右聲道音頻 .npy)
    └── mono/           (混合音頻 .npy)
```

---

## 📊 數據集統計

### 訓練集

- **樣本數**: {train_stats['total_samples']}
- **角度範圍**: [{train_stats['angle_range'][0]:.1f}°, {train_stats['angle_range'][1]:.1f}°]
- **角度平均**: {train_stats['angle_mean']:.1f}°
- **角度標準差**: {train_stats['angle_std']:.1f}°

**音頻類型分佈**:
"""
    for audio_type, count in train_stats['audio_type_distribution'].items():
        percentage = count / train_stats['total_samples'] * 100
        readme_content += f"- {audio_type}: {count} 個樣本 ({percentage:.1f}%)\n"

    readme_content += f"""
### 驗證集

- **樣本數**: {val_stats['total_samples']}
- **角度範圍**: [{val_stats['angle_range'][0]:.1f}°, {val_stats['angle_range'][1]:.1f}°]
- **角度平均**: {val_stats['angle_mean']:.1f}°
- **角度標準差**: {val_stats['angle_std']:.1f}°

**音頻類型分佈**:
"""
    for audio_type, count in val_stats['audio_type_distribution'].items():
        percentage = count / val_stats['total_samples'] * 100
        readme_content += f"- {audio_type}: {count} 個樣本 ({percentage:.1f}%)\n"

    readme_content += """
---

## 📝 metadata.json 格式

每個樣本包含以下信息：

```json
{
  "id": "train_0000",
  "left_audio": "left/train_0000.npy",
  "right_audio": "right/train_0000.npy",
  "mono_audio": "mono/train_0000.npy",
  "angle": 45.5,
  "audio_type": "speech",
  "audio_description": "男性說話聲",
  "direction_description": "右側前方大約 46 度",
  "conversation": [
    {
      "role": "user",
      "content": "請描述這段音頻的內容以及聲源的方向位置。"
    },
    {
      "role": "assistant",
      "content": "這是男性說話聲，來自右側前方大約 46 度。"
    }
  ],
  "duration": 3.0,
  "sample_rate": 16000
}
```

---

## 🔍 HRTF 模擬原理

### ITD (Interaural Time Difference)

使用 Woodworth 公式：
```
ITD = (a/c) × (sin(θ) + θ)
```
其中：
- a = 0.0875 m (頭部半徑)
- c = 343 m/s (聲速)
- θ = 角度（弧度）

### ILD (Interaural Level Difference)

簡化模型：
```
ILD = 15 × sin(θ) dB
```
並應用頻率相關的低通濾波（模擬頭部遮擋）

---

## 🔄 重新生成數據

```bash
python generate_spatial_audio_dataset.py \\
    --num_train=100 \\
    --num_val=30 \\
    --duration=3.0
```

---

## 🎯 使用方式

```python
import numpy as np
import json

# 讀取 metadata
with open('spatial_audio_data/train/metadata.json', 'r') as f:
    metadata = json.load(f)

# 載入樣本
sample = metadata[0]
left_audio = np.load(f"spatial_audio_data/train/{sample['left_audio']}")
right_audio = np.load(f"spatial_audio_data/train/{sample['right_audio']}")
angle = sample['angle']

print(f"角度: {angle}°")
print(f"對話: {sample['conversation']}")
```

---

**生成工具**: generate_spatial_audio_dataset.py
**數據格式**: NumPy (.npy) + JSON (.json)
"""

    readme_path = Path(output_dir) / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✓ README.md 已創建")


def main():
    """主函數"""
    print("=" * 80)
    print("  生成空間音頻定位數據集")
    print("=" * 80)

    output_dir = 'spatial_audio_data'

    # 生成訓練集
    train_metadata = create_dataset(
        output_dir=output_dir,
        split='train',
        num_samples=100,  # 100 個訓練樣本
        duration=3.0,
        sr=16000,
        seed=42
    )

    # 生成驗證集
    val_metadata = create_dataset(
        output_dir=output_dir,
        split='val',
        num_samples=30,  # 30 個驗證樣本
        duration=3.0,
        sr=16000,
        seed=43
    )

    # 統計信息
    train_stats = generate_statistics(train_metadata)
    val_stats = generate_statistics(val_metadata)

    print("\n" + "=" * 80)
    print("  數據集統計")
    print("=" * 80)
    print(f"\n訓練集: {train_stats['total_samples']} 個樣本")
    print(f"  角度範圍: [{train_stats['angle_range'][0]:.1f}°, {train_stats['angle_range'][1]:.1f}°]")
    print(f"  音頻類型: {train_stats['audio_type_distribution']}")

    print(f"\n驗證集: {val_stats['total_samples']} 個樣本")
    print(f"  角度範圍: [{val_stats['angle_range'][0]:.1f}°, {val_stats['angle_range'][1]:.1f}°]")
    print(f"  音頻類型: {val_stats['audio_type_distribution']}")

    # 創建 README
    create_readme(output_dir, train_stats, val_stats)

    print("\n" + "=" * 80)
    print("  ✅ 數據集生成完成！")
    print("=" * 80)
    print(f"\n數據保存在: {output_dir}/")
    print(f"  - 訓練集: {output_dir}/train/")
    print(f"  - 驗證集: {output_dir}/val/")


if __name__ == '__main__':
    main()
