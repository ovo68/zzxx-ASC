import torch
import torch.nn as nn

from model.EmbeddingLayer import EmbeddingLayer
from model.GATLayer import GATLayer
from model.GCNLayer import GCNLayer


class SSKModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        # TODO
        """
             1、固定数字都需要抽取为变量
             2、这里只是句法依赖GCN，需要换一个语义化的变量名字  √
        """

        self.syn_gcn = GCNLayer(512, 1024, 512, 0.5)
        self.com_gcn = GCNLayer(512, 1024, 512, 0.5)

        # TODO GAT

        self.sem_gat = GATLayer(512, 1024, 512, 0.2, 0.2, 8)

        # self.out = nn.Linear(in_features=512, out_features=3)
        self.out = nn.Linear(in_features=1536, out_features=3)

    def forward(self, inputs, adj1, adj2, adj3):
        lstm_outputs = self.embedding(inputs)
        # 输入句子对应的邻接矩阵：句法、常识、语义
        syn_gcn_output = self.syn_gcn(lstm_outputs, adj1)
        com_gcn_output = self.com_gcn(lstm_outputs, adj2)
        sem_gat_output = self.sem_gat(lstm_outputs, adj3)

        gcn_output = torch.cat((syn_gcn_output, com_gcn_output, sem_gat_output), dim=-1)

        out = self.out(gcn_output)
        return out
