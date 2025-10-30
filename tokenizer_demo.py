"""
Qwen2-Audio Tokenizer 使用範例
展示文字 BPE tokenizer 和音頻 feature extractor 的詳細使用方式
使用 Qwen2-Audio README 中提供的線上音檔
"""

from io import BytesIO
from urllib.request import urlopen
import librosa
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def main():
    print("=" * 80)
    print("Qwen2-Audio Tokenizer 範例 - 文字與音頻處理展示")
    print("=" * 80)

    # ============================================================
    # 1. 載入模型和 Processor
    # ============================================================
    print("\n[步驟 1] 載入模型和 Processor...")
    model_name = "Qwen/Qwen2-Audio-7B-Instruct"

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    print(f"✓ 模型載入完成: {model_name}")
    print(f"✓ Tokenizer 類型: {type(processor.tokenizer).__name__}")
    print(f"✓ Feature Extractor 類型: {type(processor.feature_extractor).__name__}")

    # ============================================================
    # 2. 文字 Tokenizer 展示 (BPE Tokenizer)
    # ============================================================
    print("\n" + "=" * 80)
    print("[步驟 2] 文字 Tokenizer 展示 (Byte-level BPE)")
    print("=" * 80)

    # 測試文字
    test_text = "請問這段音檔的內容是什麼？"

    # 使用 tokenizer 編碼文字
    text_tokens = processor.tokenizer(test_text, return_tensors="pt")

    print(f"\n原始文字: {test_text}")
    print(f"Token IDs: {text_tokens.input_ids[0].tolist()[:20]}...")  # 顯示前20個
    print(f"Token 總數: {len(text_tokens.input_ids[0])}")
    print(f"詞彙表大小: {processor.tokenizer.vocab_size:,} tokens")

    # 解碼回文字
    decoded_text = processor.tokenizer.decode(text_tokens.input_ids[0])
    print(f"解碼後文字: {decoded_text}")

    # 顯示 token 細節
    print("\n文字 Tokenization 詳細資訊:")
    tokens = processor.tokenizer.convert_ids_to_tokens(text_tokens.input_ids[0][:10])
    for i, (token_id, token) in enumerate(zip(text_tokens.input_ids[0][:10], tokens)):
        print(f"  [{i}] ID: {token_id:5d} -> Token: '{token}'")

    # ============================================================
    # 3. 音頻處理展示 (Whisper-based Feature Extractor)
    # ============================================================
    print("\n" + "=" * 80)
    print("[步驟 3] 音頻 Feature Extractor 展示 (Whisper-large-v3 based)")
    print("=" * 80)

    # 使用 README 中提供的線上音檔
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"

    print(f"\n音檔來源: {audio_url}")
    print("正在下載音檔...")

    # 載入音頻
    audio_data = urlopen(audio_url).read()
    audio, sr = librosa.load(
        BytesIO(audio_data),
        sr=processor.feature_extractor.sampling_rate
    )

    print(f"✓ 音檔載入完成")
    print(f"  - 採樣率: {processor.feature_extractor.sampling_rate} Hz")
    print(f"  - 音檔長度: {len(audio)} 個樣本")
    print(f"  - 音檔時長: {len(audio) / processor.feature_extractor.sampling_rate:.2f} 秒")

    # 音頻特徵提取
    print("\n音頻特徵提取設定:")
    print(f"  - Mel 頻譜通道數: 128 channels")
    print(f"  - 窗口大小: 25ms")
    print(f"  - Hop length: 10ms")

    # 處理音頻（需要提供文字，使用簡單的音頻標記）
    # 注意：Qwen2AudioProcessor 要求必須同時提供文字和音頻
    simple_audio_prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    audio_features = processor(
        text=simple_audio_prompt,
        audios=[audio],
        return_tensors="pt"
    )

    # 檢查返回的內容
    print(f"\n處理器返回的鍵: {audio_features.keys()}")

    if 'input_features' in audio_features:
        print(f"\n音頻特徵張量形狀: {audio_features.input_features.shape}")
        print(f"  - 維度說明: [batch_size, mel_channels, time_frames]")

        # 顯示音頻特徵的統計資訊
        print(f"\n音頻特徵統計:")
        print(f"  - 最小值: {audio_features.input_features.min().item():.4f}")
        print(f"  - 最大值: {audio_features.input_features.max().item():.4f}")
        print(f"  - 平均值: {audio_features.input_features.mean().item():.4f}")
    else:
        print("\n⚠️  注意: 音頻特徵未包含在返回中（可能是版本問題）")
        print("    將在步驟4中展示完整的音頻+文字處理")

    # ============================================================
    # 4. 完整對話範例 - 文字 + 音頻
    # ============================================================
    print("\n" + "=" * 80)
    print("[步驟 4] 完整對話範例 - 結合文字與音頻")
    print("=" * 80)

    # 構建對話
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_url},
            {"type": "text", "text": "這是什麼聲音？"},
        ]},
    ]

    # 應用聊天模板（文字 tokenization）
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    print(f"\n聊天模板處理後的文字提示:")
    print(f"{text[:200]}...")  # 顯示前200個字符

    # 提取音頻
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            BytesIO(urlopen(ele['audio_url']).read()),
                            sr=processor.feature_extractor.sampling_rate
                        )[0]
                    )

    # 統一處理文字和音頻
    print("\n正在處理多模態輸入（文字 + 音頻）...")
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)

    print(f"\n處理後的輸入:")
    print(f"  - input_ids 形狀: {inputs.input_ids.shape}")
    print(f"  - input_features 形狀: {inputs.input_features.shape if 'input_features' in inputs else 'N/A'}")
    print(f"  - attention_mask 形狀: {inputs.attention_mask.shape}")

    # 顯示特殊音頻 tokens
    print("\n特殊音頻 Tokens:")
    special_tokens = ["<|audio_bos|>", "<|AUDIO|>", "<|audio_eos|>"]
    for token in special_tokens:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID = {token_id}")

    # ============================================================
    # 5. 生成回應
    # ============================================================
    print("\n" + "=" * 80)
    print("[步驟 5] 模型生成回應")
    print("=" * 80)

    print("\n正在生成回應...")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_length=256)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]

    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(f"\n🤖 模型回應: {response}")

    # ============================================================
    # 6. 總結
    # ============================================================
    print("\n" + "=" * 80)
    print("總結 - Qwen2-Audio Tokenization 架構")
    print("=" * 80)
    print("""
    📝 文字處理:
       - Tokenizer: Byte-level BPE (來自 Qwen2)
       - 詞彙表: 151,000 tokens
       - 功能: apply_chat_template(), encode(), decode()

    🎵 音頻處理:
       - Feature Extractor: Whisper-large-v3 based
       - 採樣率: 16kHz
       - 特徵: 128-channel mel-spectrogram
       - 特殊 tokens: <|audio_bos|>, <|AUDIO|>, <|audio_eos|>

    🔗 統一處理:
       - AutoProcessor 整合兩種模態
       - processor(text=..., audios=...) 統一介面
    """)

if __name__ == "__main__":
    main()
