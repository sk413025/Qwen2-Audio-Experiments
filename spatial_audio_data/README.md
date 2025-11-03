# 空間音頻定位數據集
## 使用 HRTF 模擬生成的雙耳音頻

**生成時間**: 2025-11-03T04:29:26
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

- **樣本數**: 100
- **角度範圍**: [-84.8°, 87.8°]
- **角度平均**: -3.2°
- **角度標準差**: 49.0°

**音頻類型分佈**:
- music: 18 個樣本 (18.0%)
- speech: 43 個樣本 (43.0%)
- environmental: 39 個樣本 (39.0%)

### 驗證集

- **樣本數**: 30
- **角度範圍**: [-84.8°, 80.7°]
- **角度平均**: 7.3°
- **角度標準差**: 49.2°

**音頻類型分佈**:
- speech: 12 個樣本 (40.0%)
- music: 11 個樣本 (36.7%)
- environmental: 7 個樣本 (23.3%)

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
python generate_spatial_audio_dataset.py \
    --num_train=100 \
    --num_val=30 \
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
