import torch
from torch.utils.data import dataloader

from model.SSKGNN import SSKModel
from utils import load_data, ZXinDataset

if __name__ == '__main__':
    instances = load_data()
    train_dataset = ZXinDataset(instances)

    train_loader = dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=3,
        shuffle=True
    )

    net = SSKModel()

    for data in train_loader:
        item, label, dt = data
        # TODO 邻接矩阵
        adj_tensor = torch.randn(len(item), 80, 80)

        ssk_out = net(item, dt)
        print(ssk_out.shape)
        # print(ssk_out)
