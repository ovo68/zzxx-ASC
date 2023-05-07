import torch
from torch.utils.data import dataloader

from model.EmbeddingLayer import EmbeddingLayer
from model.GATLayer import GATLayer
from utils import load_data, ZXinDataset

if __name__ == '__main__':
    instances = load_data()

    train_dataset = ZXinDataset(instances)

    train_loader = dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=3,
        shuffle=True
    )

    net = EmbeddingLayer()

    at = torch.randn((80, 80), dtype=torch.float32)
    for data in train_loader:
        item, label, dt, kt, at = data
        print(item)
        print(label)
        print()
        lstm_output = net(item)
        print(lstm_output.shape)
        gat = GATLayer(512, 1024, 512, 0.2, 0.2, 8)
        gat_output = gat(lstm_output, at)
        print("gat_output:", gat_output.shape)
