import torch
import torch.nn as nn  # 我们通常把 torch.nn 简写为 nn
layer = nn.Linear(15, 1).to('cuda')  # 定义一个线性层，并把它放到 GPU 上
print(layer)          # 看看层的结构
print(layer.weight)   # 看看自动生成的权重矩阵
print(layer.bias)     # 看看自动生成的偏置
print(layer.weight.grad)

input_data = torch.rand(1, 15).to('cuda') # 生成随机输入
output = layer(input_data)                # 喂给网络
print(output)                             # 看看真正的计算结果

# 1. 这是一个“标量化”的小技巧，通常我们对最终的一个数值求导
# 由于 output 是一个 tensor，我们可以先求个和变成标量，或者简单地传参
# 这里为了演示简单，我们假设它是最终 Loss，直接反向传播
output.backward() 

# 2. 再次看看权重的梯度
print(layer.weight.grad)
print(input_data)


import torch.optim as optim  # 1. 导入工具包

# 2. 定义优化器
# 括号里的第一个参数：告诉它要更新谁 (layer.parameters())
# 括号里的第二个参数：告诉它学习率是多少 (lr=0.01)
optimizer = optim.SGD(layer.parameters(), lr=0.01)
print(optimizer)