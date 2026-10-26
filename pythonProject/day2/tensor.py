# author: jkisme

import torch
# 张量的创建
t1 = torch.tensor([1,2,3,4])
print(f"t1:{t1},t1的类型:{t1.dtype}")

# 创建等差数列
t2 = torch.arange(0,10,2)
print("t2:",t2,"t2.dtype:",t2.dtype)

# 创建等间距数列
t3 = torch.linspace(0,1,5)
print("t3:",t3,"t3.dtype:",t3.dtype)

torch.manual_seed(10)
t4 = torch.randn(4,5)
print("t4:",t4,"t4.dtype:",t4.dtype)

t5 = t2.type(torch.float64)
print("t5:",t5,"t5.dtype:",t5   .dtype)

t6 = t2.double()
print("t6:", t6, "t6.dtype:", t6.dtype)

