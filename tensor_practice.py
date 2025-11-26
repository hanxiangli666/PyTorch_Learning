import torch  # 先导入工具包

x = torch.rand(5, 3) # 创建张量
print(x)             # 打印出来看看

print(torch.cuda.is_available())  # 检查 GPU 是否可用
x = x.to('cuda')
print(x.device)                   # 检查 x 目前在哪个设备上


y = torch.ones(5, 3).to('cuda')   # 直接创建在 GPU 上的张量
print(x + y)                      # 张量运算

new_x = x.view(1, 15)          # 改变张量形状
print(new_x.shape)             # 打印新形状