import torch.nn as nn

from model.EmbeddingLayer import EmbeddingLayer


class SSKModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        self.out = nn.Linear(in_features=512, out_features=3)

    def forward(self, inputs):
        lstm_output = self.embedding(inputs)

        out = self.out(lstm_output)
        return out
