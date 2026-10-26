# author: jkisme

import torch
import numpy as np

n1=np.arange(12).reshape(3,4)
t1=torch.from_numpy(n1)
print("t1:",t1)

print("t1[1,:]:",t1[1,:])
print("t1[:,1]:",t1[:,1])
print("t1[1,1]:",t1[1,1])
print("t1[1:2,1:2]:",t1[1:2,1:2])
print("t1[t1[:,2]>5]:",t1[t1[:,2]>5])
print("t1[t1[:,2]>5,:]:",t1[t1[:,2]>5,:])
print("t1[t1[:,2]>5,1:3]:",t1[t1[:,2]>5,1:3])
print("t1[t1[:,2]>5,1]:",t1[t1[:,2]>5,1])

t2=t1 * t1
print("t2:",t2)
# print("t1*t1:",t1*t1)

print("cat:",torch.cat([t1,t2],dim=0))

# 修改形状
t3 = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12])
print("view:",t3.view(2,2,3))
print("reshapge:",torch.reshape(t3,(2,2,3)))

tensor_3d = torch.arange(24).view(2, 3, 4)
print("tensor_3d:",tensor_3d)
# 2,3,4 => 3,2,4
print(tensor_3d.transpose(0,1))
# 2,3,4 => 2,4,3
print(tensor_3d.transpose(1,2))

print("t3:",t3)
print("t3.unsqueeze(0):",t3.unsqueeze(0))   # 增加层数,层数+1
print("t3.unsqueeze(1):",t3.unsqueeze(1)) # 增加维度,按行做拆分
# print("t3.unsqueeze(2):",t3.unsqueeze(2))
