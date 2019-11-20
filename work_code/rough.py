import torch
import pdb
import numpy as np

x = torch.tensor([]).cuda()
y = torch.randn((2,5)).cuda()
x = torch.cat((x, y), 0)
perm = np.random.permutation(10)

a = np.array([[1,2,3],[4,5,6]])
print(a.reshape((1,-1)))
pdb.set_trace()
print(perm)