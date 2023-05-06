import torch
import torch.nn as nn

from model.EmbeddingLayer import EmbeddingLayer
from model.GCNLayer import GCN


class SSKModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        # TODO
        """
             1、固定数字都需要抽取为变量
             2、这里只是句法依赖GCN，需要换一个语义化的变量名字  √
        """

        self.syn_gcn = GCN(512, 1024, 512, 0.5)
        self.com_gcn = GCN(512, 1024, 512, 0.5)

        # TODO GAT

        # self.out = nn.Linear(in_features=512, out_features=3)
        self.out = nn.Linear(in_features=1024, out_features=3)

    def forward(self, inputs, adj1, adj2, adj3):
        lstm_outputs = self.embedding(inputs)
        # 输入句子对应的邻接矩阵：句法、常识
        syn_gcn_output = self.syn_gcn(lstm_outputs, adj1)
        sem_gcn_output = self.com_gcn(lstm_outputs, adj2)

        gcn_output = torch.cat((syn_gcn_output, sem_gcn_output), dim=-1)

        out = self.out(gcn_output)
        return out
