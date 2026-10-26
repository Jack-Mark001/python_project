
import numpy as np

arr1=np.arange(5)
print("arr1",arr1)

arr2=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
print("arr2",arr2)
print(arr2.ndim,arr2.shape,arr2.size)

print("n1,axis=0算总和",arr2.sum(axis=0))
print("axis=1的总和:",arr2.sum(axis=1))

print("列统计:",arr2.min(axis=0),arr2.max(axis=0),arr2.mean(axis=0))

print("行统计:",arr2.min(axis=1),arr2.max(axis=1),arr2.mean(axis=1))

