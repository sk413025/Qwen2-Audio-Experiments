"""
Spatial Feature Extractor

從雙聲道 Mel-Spectrogram 提取 ITD/ILD 特徵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ITDBranch(nn.Module):
    """
    ITD (Interaural Time Difference) 特徵提取分支

    使用時間卷積捕獲左右聲道的時間對齊模式
    """
    def __init__(self, n_mels=128):
        super().__init__()

        # 輸入: 左右聲道拼接 [batch, 2*n_mels, T]

        # Layer 1: 短時相關性 (50ms)
        self.conv1 = nn.Conv1d(
            in_channels=2 * n_mels,
            out_channels=256,
            kernel_size=5,
            padding=2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(256)

        # Layer 2: 中時相關性 (90ms)
        self.conv2 = nn.Conv1d(
            in_channels=256,
            out_channels=128,
            kernel_size=9,
            padding=4,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(128)

        # Layer 3: 長時相關性 (170ms)
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=17,
            padding=8,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(64)

        # 激活函數
        self.relu = nn.ReLU()

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, left_mel, right_mel):
        """
        Args:
            left_mel: [batch, n_mels, T]
            right_mel: [batch, n_mels, T]

        Returns:
            itd_features: [batch, 64]
        """
        # 拼接左右聲道
        x = torch.cat([left_mel, right_mel], dim=1)  # [batch, 2*n_mels, T]

        # 多尺度時間卷積
        x = self.relu(self.bn1(self.conv1(x)))  # [batch, 256, T]
        x = self.relu(self.bn2(self.conv2(x)))  # [batch, 128, T]
        x = self.relu(self.bn3(self.conv3(x)))  # [batch, 64, T]

        # 全局平均池化
        x = self.gap(x).squeeze(-1)  # [batch, 64]

        return x


class ILDBranch(nn.Module):
    """
    ILD (Interaural Level Difference) 特徵提取分支

    提取頻率相關的能量差異
    """
    def __init__(self, n_mels=128):
        super().__init__()

        # 頻率加權（高頻更重要）
        freq_weight = torch.linspace(0.5, 1.5, n_mels)
        self.register_buffer('freq_weight', freq_weight)

        # 能量差異處理
        self.conv1 = nn.Conv1d(
            in_channels=n_mels,
            out_channels=128,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, left_mel, right_mel):
        """
        Args:
            left_mel: [batch, n_mels, T]
            right_mel: [batch, n_mels, T]

        Returns:
            ild_features: [batch, 64]
        """
        # 計算能量差異
        energy_diff = left_mel - right_mel  # [batch, n_mels, T]

        # 頻率加權
        freq_weight = self.freq_weight.view(1, -1, 1)  # [1, n_mels, 1]
        weighted_diff = energy_diff * freq_weight

        # 卷積處理
        x = self.relu(self.bn1(self.conv1(weighted_diff)))  # [batch, 128, T]
        x = self.relu(self.bn2(self.conv2(x)))              # [batch, 64, T]

        # 全局平均池化
        x = self.gap(x).squeeze(-1)  # [batch, 64]

        return x


class SpatialFeatureExtractor(nn.Module):
    """
    完整的空間特徵提取器

    結合 ITD 和 ILD 分支提取空間方向特徵
    """
    def __init__(self, n_mels=128, output_dim=256):
        super().__init__()

        # ITD 分支
        self.itd_branch = ITDBranch(n_mels)

        # ILD 分支
        self.ild_branch = ILDBranch(n_mels)

        # 融合層
        self.fusion = nn.Sequential(
            nn.Linear(128, 256),  # 64 (ITD) + 64 (ILD) = 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )

    def forward(self, left_mel, right_mel):
        """
        Args:
            left_mel: [batch, n_mels, T] 左聲道 Mel-Spectrogram
            right_mel: [batch, n_mels, T] 右聲道 Mel-Spectrogram

        Returns:
            spatial_features: [batch, output_dim] 空間方向特徵
        """
        # 提取 ITD 特徵
        itd_features = self.itd_branch(left_mel, right_mel)  # [batch, 64]

        # 提取 ILD 特徵
        ild_features = self.ild_branch(left_mel, right_mel)  # [batch, 64]

        # 拼接並融合
        combined = torch.cat([itd_features, ild_features], dim=1)  # [batch, 128]
        spatial_features = self.fusion(combined)  # [batch, output_dim]

        return spatial_features


def test_spatial_extractor():
    """測試 Spatial Feature Extractor"""
    print("=" * 80)
    print("  測試 Spatial Feature Extractor")
    print("=" * 80)

    # 創建模型
    model = SpatialFeatureExtractor(n_mels=128, output_dim=256)

    # 統計參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型參數:")
    print(f"  總參數: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}")

    # 測試前向傳播
    batch_size = 4
    n_mels = 128
    seq_len = 100

    left_mel = torch.randn(batch_size, n_mels, seq_len)
    right_mel = torch.randn(batch_size, n_mels, seq_len)

    print(f"\n輸入形狀:")
    print(f"  Left Mel-Spec: {left_mel.shape}")
    print(f"  Right Mel-Spec: {right_mel.shape}")

    # 前向傳播
    model.eval()
    with torch.no_grad():
        spatial_features = model(left_mel, right_mel)

    print(f"\n輸出形狀:")
    print(f"  Spatial Features: {spatial_features.shape}")

    print(f"\n特徵統計:")
    print(f"  Mean: {spatial_features.mean():.4f}")
    print(f"  Std: {spatial_features.std():.4f}")
    print(f"  Min: {spatial_features.min():.4f}")
    print(f"  Max: {spatial_features.max():.4f}")

    print("\n✅ 測試通過！")


if __name__ == '__main__':
    test_spatial_extractor()
