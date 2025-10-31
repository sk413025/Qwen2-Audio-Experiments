"""
Qwen2-Audio Stage 1 训练 - 梯度流测试成功版本

关键发现：
=========
问题根源：使用了错误的 processor 参数名
- ❌ 错误：processor(text=text, audios=[audio])  # 'audios' (复数)
- ✅ 正确：processor(text=text, audio=audio)     # 'audio' (单数)

使用 'audios' 时：
  - processor 会忽略该参数（有警告信息）
  - 返回的 inputs 中没有 'input_features'（音频特征）
  - 模型只处理文本，audio_tower 和 projector 不会被调用
  - 自然没有梯度流到音频组件

使用 'audio' 时：
  - processor 正确处理音频数据
  - 返回的 inputs 包含 'input_features' 和 'feature_attention_mask'
  - 模型正常处理音频，调用 audio_tower 和 projector
  - 梯度正确流到音频组件 ✅

训练策略：
=========
Stage 1 目标：训练音频组件，冻结语言模型
- audio_tower (Qwen2AudioEncoder): 可训练 ✅
- multi_modal_projector (Audio Adapter): 可训练 ✅
- language_model (Qwen2ForCausalLM): 冻结 ❄️

实现方式：
- 只需将 language_model 的所有参数设为 requires_grad=False
- audio_tower 和 multi_modal_projector 保持可训练
- 梯度会正确地只流向未冻结的组件
"""

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import numpy as np

print("=" * 100)
print("  Qwen2-Audio Stage 1 梯度流测试 - 成功版本")
print("=" * 100)

MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

# ============================================================================
# 1. 加载模型和处理器
# ============================================================================
print("\n1. 加载模型和处理器")
print("-" * 100)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map='cpu',
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print(f"✓ 模型加载完成")
print(f"  总参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# ============================================================================
# 2. 应用 Stage 1 训练策略：只冻结 language_model
# ============================================================================
print("\n2. 应用 Stage 1 训练策略")
print("-" * 100)

# 冻结 language_model（包括 embedding、transformer layers、lm_head）
for param in model.language_model.parameters():
    param.requires_grad = False

# 验证冻结策略
audio_tower_trainable = sum(p.numel() for p in model.audio_tower.parameters() if p.requires_grad)
projector_trainable = sum(p.numel() for p in model.multi_modal_projector.parameters() if p.requires_grad)
llm_trainable = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)

print(f"参数状态:")
print(f"  audio_tower: {audio_tower_trainable / 1e6:.2f}M 可训练")
print(f"  multi_modal_projector: {projector_trainable / 1e6:.2f}M 可训练")
print(f"  language_model: {llm_trainable / 1e6:.2f}M 可训练 (应为 0)")

assert llm_trainable == 0, "language_model 应该完全冻结"
print(f"✓ 冻结策略正确")

# ============================================================================
# 3. 准备训练数据
# ============================================================================
print("\n3. 准备训练数据")
print("-" * 100)

# 构建多模态对话
conversation = [
    {
        'role': 'user',
        'content': [
            {'type': 'audio', 'audio_url': 'test.wav'},
            {'type': 'text', 'text': '请转录'}
        ]
    },
    {
        'role': 'assistant',
        'content': '今天天气真好'
    }
]

# 应用聊天模板
text = processor.apply_chat_template(
    conversation,
    add_generation_prompt=False,
    tokenize=False
)

print(f"生成的文本模板:")
print(f"  长度: {len(text)} 字符")
print(f"  包含 <|AUDIO|> token: {'<|AUDIO|>' in text}")

# 生成模拟音频（3秒，16kHz采样率）
audio = np.random.randn(16000 * 3).astype(np.float32)
print(f"\n音频数据:")
print(f"  形状: {audio.shape}")
print(f"  采样率: 16000 Hz")
print(f"  时长: 3.0 秒")

# ============================================================================
# 4. 关键步骤：使用正确的参数名调用 processor
# ============================================================================
print("\n4. 使用 processor 处理输入")
print("-" * 100)

# ✅ 正确方式：使用 'audio' (单数)
print("调用方式: processor(text=text, audio=audio, return_tensors='pt')")
inputs = processor(
    text=text,
    audio=audio,  # ✅ 关键：使用 'audio' 而不是 'audios'
    return_tensors='pt'
)

print(f"\n返回的 keys:")
for key in inputs.keys():
    if torch.is_tensor(inputs[key]):
        print(f"  {key}: shape={inputs[key].shape}")

# 验证音频特征是否存在
if 'input_features' in inputs:
    print(f"\n✓ 成功！音频特征已包含在 inputs 中")
    print(f"  input_features shape: {inputs['input_features'].shape}")
else:
    print(f"\n✗ 警告：inputs 中没有 input_features")
    print(f"  这意味着音频数据没有被处理")

# ============================================================================
# 5. Forward Pass
# ============================================================================
print("\n5. Forward Pass")
print("-" * 100)

# 准备标签（用于计算损失）
labels = inputs['input_ids'].clone()

# 设置为训练模式
model.train()

# Forward
outputs = model(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    input_features=inputs.get('input_features'),
    feature_attention_mask=inputs.get('feature_attention_mask'),
    labels=labels
)

print(f"Loss:")
print(f"  值: {outputs.loss.item():.4f}")
print(f"  requires_grad: {outputs.loss.requires_grad}")
print(f"  grad_fn: {outputs.loss.grad_fn}")

# ============================================================================
# 6. Backward Pass
# ============================================================================
print("\n6. Backward Pass")
print("-" * 100)

if outputs.loss.requires_grad:
    outputs.loss.backward()
    print("✓ Backward 完成")
else:
    print("✗ 错误：loss 不可求导")
    exit(1)

# ============================================================================
# 7. 验证梯度分布
# ============================================================================
print("\n7. 验证梯度分布")
print("-" * 100)

# 统计每个组件的梯度情况
def count_gradients(module, name):
    """统计模块中有梯度的参数数量"""
    total_params = 0
    params_with_grad = 0
    params_with_nonzero_grad = 0

    for param in module.parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            if param.grad.abs().sum() > 0:
                params_with_nonzero_grad += 1

    return total_params, params_with_grad, params_with_nonzero_grad

audio_stats = count_gradients(model.audio_tower, 'audio_tower')
proj_stats = count_gradients(model.multi_modal_projector, 'multi_modal_projector')
llm_stats = count_gradients(model.language_model, 'language_model')

print(f"梯度统计:")
print(f"\naudio_tower:")
print(f"  总参数: {audio_stats[0]}")
print(f"  有梯度: {audio_stats[1]}")
print(f"  非零梯度: {audio_stats[2]}")

print(f"\nmulti_modal_projector:")
print(f"  总参数: {proj_stats[0]}")
print(f"  有梯度: {proj_stats[1]}")
print(f"  非零梯度: {proj_stats[2]}")

print(f"\nlanguage_model:")
print(f"  总参数: {llm_stats[0]}")
print(f"  有梯度: {llm_stats[1]}")
print(f"  非零梯度: {llm_stats[2]}")

# ============================================================================
# 8. 最终验证
# ============================================================================
print("\n" + "=" * 100)
print("  最终验证")
print("=" * 100)

success = True
errors = []

# 检查 1: audio_tower 应该有梯度
if audio_stats[2] == 0:
    success = False
    errors.append("audio_tower 没有梯度")
else:
    print(f"✓ audio_tower 有梯度 ({audio_stats[2]}/{audio_stats[0]} 参数)")

# 检查 2: multi_modal_projector 应该有梯度
if proj_stats[2] == 0:
    success = False
    errors.append("multi_modal_projector 没有梯度")
else:
    print(f"✓ multi_modal_projector 有梯度 ({proj_stats[2]}/{proj_stats[0]} 参数)")

# 检查 3: language_model 不应该有梯度
if llm_stats[2] > 0:
    success = False
    errors.append(f"language_model 有梯度 ({llm_stats[2]} 参数)")
else:
    print(f"✓ language_model 没有梯度 (正确冻结)")

print("\n" + "=" * 100)
if success:
    print("🎉🎉🎉 成功！Stage 1 训练策略验证通过")
    print("=" * 100)
    print("\n关键要点总结:")
    print("1. 使用正确的参数名: audio (单数) 而不是 audios (复数)")
    print("2. 只冻结 language_model，保持 audio_tower 和 projector 可训练")
    print("3. 梯度正确流向音频组件，不会流向冻结的语言模型")
    print("4. 这就是 Qwen2-Audio Stage 1 训练的正确方式")
else:
    print("❌ 验证失败")
    print("=" * 100)
    for error in errors:
        print(f"  - {error}")
    exit(1)

print("\n" + "=" * 100)
print("  调查历程回顾")
print("=" * 100)

print("""
问题发现过程：
1. 初始问题：RuntimeError: element 0 of tensors does not require grad
2. 尝试多种冻结策略：全部失败
3. 测试简化模型：成功 → 说明 PyTorch 机制没问题
4. 深入 transformers 源码：masked_scatter 支持梯度
5. 检查 processor 输入：发现使用了错误的参数名 'audios'
6. 修正为 'audio'：问题解决 ✅

教训：
- 仔细检查 API 文档和参数名
- 使用错误的参数名可能导致数据被静默忽略
- 当模型行为异常时，检查输入数据是否正确传递
- 简化模型测试是排查问题的有效方法
""")

print("=" * 100)
