import json
import torch

# 加载模型
model_path = '/opt/HuaZang/Baichuan2-13B-Chat/pytorch_model-00001-of-00003.bin'
model_state_dict = torch.load(model_path, map_location='cpu')

# 获取每个层的权重
for layer_name, weights in model_state_dict.items():
    print(f"Layer: {layer_name}")
    print(f"Weights: {weights}", weights.shape)

