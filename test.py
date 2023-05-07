import torch

x = torch.randn((3, 80, 6))
y = torch.randn((3, 80, 6))


z1 = torch.cat([x,y],dim=1)
z2 = torch.cat([x,y],dim=2)
print(z1.shape)
print(z2.shape)
