import numpy
import time

data1=list(range(1000000))
start_time=time.time()
result_list=[x**2 for x in data1]
t1=time.time()-start_time
print("python耗时:",t1)

# numpy
data2=list(range(1000000))
start_time=time.time()

result_array=numpy.array(data2)**2
t2=time.time()-start_time
print("numpy耗时:",t2)

# 倍率
print("倍率:",t1/t2)


