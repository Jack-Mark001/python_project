# author:jkisme

import numpy as np

arr1=np.array(range(12))
print("arr1",arr1)
print("一维操作--------------")
print(arr1[1])
print(arr1[-1])  # -1是指获取最后一个元素
print(arr1[1:4])  # 从下标1 取到 下标4
print(arr1[:4])   # 取4前面的值  不包含4
print(arr1 [1:])   # 取2这个位置之后的

print("二维操作--------------")
arr2=arr1.reshape(3,4)
print("arr2",arr2)
print(arr2[1,2])
print(arr2[1:3,2:4])
print(arr2[1:3])
print(arr2[1:3,:])
print(arr2[:,1:3])
print(arr2[1:3,1:3])


print("数组的拼接--------------")
arr3=np.array([[1,2,3],[4,5,6]])
print("arr3",arr3)

arr4=np.array([[7,8,9],[10,11,12]])
print("arr4",arr4)
arr5=np.concatenate((arr3,arr4),axis=0)
print("arr5",arr5)
arr6=np.concatenate((arr3,arr4),axis=1)
print("arr6",arr6)

arr7=np.vstack((arr3,arr4))
print("arr7",arr7)

arr8=np.hstack((arr3,arr4))
print("arr8",arr8)

print("数组的分割--------------")
arr9=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("arr9",arr9)

arr11=np.split(arr9,2,axis=1)
print("arr11",arr11)
arr12=np.array_split(arr9,2,axis=1)
print("arr12",arr12)


v1 = [[0,1,2,3,4,5],
      [6,7,8,9,10,11]]  # 2x6数组
v2 = [[12,13,14,15,16,17],
      [18,19,20,21,22,23]]  # 2x6数组

v1 = np.vstack((arr1,arr2))
print("v1",v1)
# v4 =>  concatenate(axis=1)
v2 = np.hstack((arr1,arr2))
print("v2",v2)

arr = np.arange(9).reshape(3,3)
print("arr",arr)

# 分割
arr12 = np.split(arr,3,axis=1)
print("arr12",arr12)

arr12 = np.split(arr,3,axis=0)
print("arr12",arr12)
# axis=1
print("hsplit",np.hsplit(arr12,3))
# axis=0 按行做分割
print("vsplit",np.vsplit(arr12,3))


















