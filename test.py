import numpy as np


temp_arr = np.zeros((10,2))

for index in range(10):
    temp_arr[index,0] = index*3
    temp_arr[index,1] = index-2

print(temp_arr)
mean_ij = np.mean(temp_arr, axis=0)


print(mean_ij)
np.linalg.norm(mean_ij)