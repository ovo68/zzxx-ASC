import torch

x = torch.tensor([2])
print(x.shape)
x = torch.unsqueeze(x, dim=0)
print(x.shape)
