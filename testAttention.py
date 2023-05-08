import torch
from torch.utils.data import dataloader

from model.Attention import ScaledDotProductAttention
from model.EmbeddingLayer import EmbeddingLayer
from utils import ZXinDataset
from utils import load_data

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
        # print(item)
        # print(label)
        # print()
        lstm_output = net(item)
        # print(lstm_output.shape)
        sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, d_out=80, h=2)
        sa_output = sa(lstm_output, lstm_output, lstm_output)
        print(sa_output.shape)
