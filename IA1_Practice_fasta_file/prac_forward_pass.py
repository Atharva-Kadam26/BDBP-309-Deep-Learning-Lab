# import torch
# import torch.nn as nn
#
# class SingleLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(in_features=3,out_features=1)
#
#     def forward(self,x):
#         output = self.linear(x)
#         return output
#
# model = SingleLayer()
# x = torch.tensor([[1.0,2.0,3.0]])
#
# output = model(x)
# print("Input:",x)
# print("Output:",output)


import torch
from torch import nn

class SingleLayer(nn.Module):
    def __init__(self):
        super(SingleLayer, self).__init__()
        self.layer = nn.Linear(
            in_features = 3,
            out_features = 1,
        )

    def forward(self, x):
        x = self.layer(x)
        return x

model = SingleLayer()
x = torch.tensor([[1.0,2.0,3.0]])

output = model(x)
print("Input:",x)
print("Output:",output)