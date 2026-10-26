# author: jkisme
import numpy as np

# nan
a=np.nan
print("a=",a)

arr = np.array([1,2,3,np.nan,np.inf,4,5,np.nan,0,0])
print("arr:",arr)

arr2 = np.where(np.isinf(arr),0,arr)
print("arr2:",arr2)


print("统计个数:",np.count_nonzero(arr))

# 求和操作
arr3 = np.array([1,2,3,np.nan,5,6,np.nan])
print("sum:",np.sum(arr3))
print("nansum:",np.nansum(arr3))