from torch.utils.data import dataloader

from model.EmbeddingLayer import EmbeddingLayer
from utils import load_data, ZXinDataset

if __name__ == '__main__':
    instances = load_data()
    for i in instances:
        print(i)

    print()

    train_dataset = ZXinDataset(instances)

    train_loader = dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=3,
        shuffle=True
    )

    net = EmbeddingLayer()

    for data in train_loader:
        item, label = data
        print(item)
        print(label)
        print()
        lstm_output = net(item)
        print(lstm_output.shape)
