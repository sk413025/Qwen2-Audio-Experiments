"""
檢查並顯示 Qwen2-Audio 的文字 tokens 和音頻特徵
"""

from io import BytesIO
from urllib.request import urlopen
import librosa
import torch
import numpy as np
from transformers import AutoProcessor

def main():
    print("=" * 80)
    print("檢查 Qwen2-Audio 的 Tokens 和特徵")
    print("=" * 80)

    # 載入 processor (不需要載入整個模型)
    print("\n載入 Processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    print("✓ 完成")

    # ============================================================
    # 1. 文字 Tokens
    # ============================================================
    print("\n" + "=" * 80)
    print("1. 文字 Tokenization - 產生離散的 Token IDs")
    print("=" * 80)

    test_texts = [
        "你好",
        "這是一段測試文字",
        "請問這段音檔的內容是什麼？"
    ]

    for text in test_texts:
        tokens = processor.tokenizer(text, return_tensors="pt")
        token_ids = tokens.input_ids[0].tolist()

        print(f"\n文字: {text}")
        print(f"Token IDs: {token_ids}")
        print(f"Token 數量: {len(token_ids)}")

        # 顯示每個 token 的詳細資訊
        token_strs = processor.tokenizer.convert_ids_to_tokens(tokens.input_ids[0])
        print("Token 詳細:")
        for i, (tid, tstr) in enumerate(zip(token_ids, token_strs)):
            print(f"  [{i}] ID={tid:6d} → '{tstr}'")

    # ============================================================
    # 2. 音頻特徵
    # ============================================================
    print("\n" + "=" * 80)
    print("2. 音頻 Feature Extraction - 產生連續的特徵向量")
    print("=" * 80)

    # 載入音頻
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
    print(f"\n下載音檔: {audio_url}")
    audio_data = urlopen(audio_url).read()
    audio, sr = librosa.load(
        BytesIO(audio_data),
        sr=processor.feature_extractor.sampling_rate
    )

    print(f"✓ 音檔資訊:")
    print(f"  - 採樣率: {sr} Hz")
    print(f"  - 樣本數: {len(audio)}")
    print(f"  - 時長: {len(audio)/sr:.2f} 秒")
    print(f"  - 數據類型: {audio.dtype}")
    print(f"  - 數值範圍: [{audio.min():.4f}, {audio.max():.4f}]")

    # 使用 feature_extractor 直接處理音頻
    print("\n提取音頻特徵...")
    audio_features = processor.feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    print(f"✓ 音頻特徵:")
    print(f"  - 特徵張量形狀: {audio_features.input_features.shape}")
    print(f"    [batch_size, mel_channels, time_frames]")
    print(f"    = [1, {audio_features.input_features.shape[1]}, {audio_features.input_features.shape[2]}]")
    print(f"  - 數據類型: {audio_features.input_features.dtype}")
    print(f"  - 數值範圍: [{audio_features.input_features.min():.4f}, {audio_features.input_features.max():.4f}]")
    print(f"  - 平均值: {audio_features.input_features.mean():.4f}")
    print(f"  - 標準差: {audio_features.input_features.std():.4f}")

    # 顯示特徵的一小部分
    print("\n特徵矩陣的前 5x5 數值:")
    feature_sample = audio_features.input_features[0, :5, :5].numpy()
    for i in range(5):
        row_str = "  " + " ".join([f"{val:7.3f}" for val in feature_sample[i]])
        print(row_str)

    # ============================================================
    # 3. 特殊音頻 Tokens
    # ============================================================
    print("\n" + "=" * 80)
    print("3. 特殊音頻 Tokens (用於標記音頻位置)")
    print("=" * 80)

    special_tokens = {
        "<|audio_bos|>": "音頻開始標記",
        "<|AUDIO|>": "音頻佔位符",
        "<|audio_eos|>": "音頻結束標記"
    }

    print("\n這些是特殊的離散 tokens，用於在文字序列中標記音頻位置:")
    for token, desc in special_tokens.items():
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        print(f"  {token:20s} ID={token_id:6d}  ({desc})")

    # 顯示一個包含音頻標記的完整文字序列
    print("\n範例：包含音頻標記的文字序列:")
    audio_text = "<|audio_bos|><|AUDIO|><|audio_eos|>這是什麼聲音？"
    tokens = processor.tokenizer(audio_text, return_tensors="pt")
    token_ids = tokens.input_ids[0].tolist()
    token_strs = processor.tokenizer.convert_ids_to_tokens(tokens.input_ids[0])

    print(f"\n文字: {audio_text}")
    print("Token 序列:")
    for i, (tid, tstr) in enumerate(zip(token_ids, token_strs)):
        marker = " ← 音頻標記" if tstr in special_tokens else ""
        print(f"  [{i:2d}] ID={tid:6d} → '{tstr}'{marker}")

    # ============================================================
    # 總結
    # ============================================================
    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)
    print("""
    文字處理:
      輸入: 字符串 (例如: "你好")
      輸出: 離散的 Token IDs (例如: [104387, 105515])
      特點: 每個 token 是一個整數，可以查表轉換回子詞

    音頻處理:
      輸入: 音頻波形數組 (numpy array)
      輸出: 連續的特徵矩陣 (mel-spectrogram)
      特點: 128 個 mel-frequency 通道的連續浮點數值

    在模型中的使用:
      - 文字 tokens 和特殊音頻 tokens 組成輸入序列
      - 音頻特徵通過 audio encoder 處理
      - 兩者在模型內部融合進行多模態理解
    """)

if __name__ == "__main__":
    main()
