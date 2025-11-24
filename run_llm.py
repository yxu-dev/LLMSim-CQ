import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset

# ds = load_dataset("openai/gsm8k", "main")

# --- 1. 设置模型 ---
# 确保你已经通过 `huggingface-cli login` 登录
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# 加载分词器和模型
# device_map="auto" 会自动将模型加载到可用的GPU上，极大提高速度
# torch_dtype="auto" 会自动选择最佳的数据类型（如 bfloat16）以优化显存和性能
print(f"正在加载模型: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)
print("模型加载完成！")
print("-" * 50)

# --- 2. 访问权重 (Weights) ---
# 权重是模型训练好的参数，我们可以直接访问它们。
# model.named_parameters() 会返回一个包含(参数名, 参数张量)的迭代器。

print("访问模型权重 (示例):")
# 为了不打印所有上百个参数，我们只看前5个
count = 0
for name, param in model.named_parameters():
    print(f"层名称: {name}")
    print(f"权重形状: {param.shape}")
    print(f"需要梯度: {param.requires_grad}")
    print("-" * 20)
    count += 1
    if count >= 5:
        break
print("权重示例展示完毕。")
print("-" * 50)


# --- 3. 捕获激活值 (Activations) ---
# 激活值是在前向传播过程中动态生成的，需要用“钩子”来捕获。
# 我们将捕获每个解码器层 (Decoder Layer) 的输出激活。

print("设置钩子以捕获激活值...")

# 创建一个字典来存储我们捕获的激活值
# 键是层的名称，值是该层的输出张量
activations = {}

# 定义钩子函数
# 这个函数将在每个我们注册了钩子的模块完成前向传播后被调用
def get_activation(name):
    def hook(model, input, output):
        # Llama模型的输出是一个元组，我们通常关心第一个元素，即hidden_states
        activations[name] = output[0].detach()
    return hook

# 注册钩子
# Llama 3.1的模型结构中，所有Transformer块都存储在 model.model.layers 中
# 我们遍历这些层，并为每一个层注册一个前向钩子
hooks = []
for layer_idx, layer in enumerate(model.model.layers):
    layer_name = f"decoder_layer_{layer_idx}"
    hook_handle = layer.register_forward_hook(get_activation(layer_name))
    hooks.append(hook_handle) # 保存句柄，以便之后移除钩子

print(f"已在 {len(hooks)} 个解码器层上注册了钩子。")
print("-" * 50)


# --- 4. 运行前向传播以触发钩子 ---
# 只有当数据流过模型时，钩子才会被触发，激活值才会被捕获

print("准备输入并执行前向传播...")
# 准备一个输入样本
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 执行前向传播
# 我们不需要计算梯度，所以使用 torch.no_grad() 来节省计算资源
with torch.no_grad():
    outputs = model(**inputs)

print("前向传播完成！")
print("-" * 50)


# --- 5. 检查捕获到的激活值 ---

print("检查已捕获的激活值:")
if activations:
    for name, activation_tensor in activations.items():
        print(f"层名称: {name}")
        # 激活值的形状通常是 [batch_size, sequence_length, hidden_dim]
        print(f"激活值形状: {activation_tensor.shape}")
        print("-" * 20)
else:
    print("未能捕获到任何激活值。")

print("-" * 50)


# --- 6. (可选) 移除钩子 ---
# 在完成分析后，移除钩子是一个好习惯，以防它们影响后续操作
print("移除所有钩子...")
for hook_handle in hooks:
    hook_handle.remove()
print("钩子已移除。")