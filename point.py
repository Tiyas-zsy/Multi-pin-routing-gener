# point.py

import sys

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = sys.maxsize
        self.parent =None
class Point_arr:
    def __init__(self,input):
        self.x = input[0]
        self.y = input[1]
        self.cost = sys.maxsize



'''# 创建一个三维数组
import numpy as np
arr = np.array([[0, 1], [1, 0],[1, 0],[1, 0], 
               [1, 1], [0, 1],[1, 0],[1, 0]])
print(arr.shape)
print(len(arr),len(arr[0]))
# 找到值为1的元素的索引
indices = np.argwhere(arr[:,:,1] == 1)

# 打印结果
print(indices)'''