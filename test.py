import matplotlib.pyplot as plt
import torch
import numpy as np


tensor = torch.randn(4,2)
print(tensor)
# 对张量进行操作
result_tensor = torch.where(tensor > 0.5, tensor + 0.5,tensor - 0.5)
print(result_tensor)