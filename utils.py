import numpy as np
import spacy
import torch
from pandas import read_csv
from torch.utils.data import dataset


class Instance:
    def __init__(self, no, sentence, aspect, polarity, dt, x, label) -> None:
        super().__init__()
        self.target = None
        self.no = no
        self.sentence = sentence
        self.aspect = aspect
        self.polarity = polarity
        self.dt = dt
        """
            x = ([CLS] s [SEP] a [SEP])
            x = (s [SEP] a)
        """
        self.x = x
        self.label = label

    def __str__(self) -> str:
        return 'no:' + str(
            self.no) + ',sentence=' + self.sentence + ', aspect=' + self.aspect + ',polarity=' + self.polarity + ',x=' + self.x


def load_data(path='demo.csv'):
    f = open(path, encoding='UTF-8')
    column_names = ['id', 'Sentence', 'Aspect_Term', 'polarity', 'from', 'to']
    origin_data = read_csv(f, names=column_names, header=None, skiprows=1)

    instances_train = []
    origin_data_list = [origin_data[column_names[i]].tolist() for i in range(len(column_names))]
    for i in range(len(origin_data_list[0])):
        _, dt, _ = adj_dependency_tree(origin_data_list[1][i])
        instances_train.append(Instance(
            origin_data_list[0][i],
            origin_data_list[1][i],
            origin_data_list[2][i],
            origin_data_list[3][i],
            torch.tensor(dt, dtype=torch.float32),
            # '[CLS]' + origin_data_list[1][i] + '[SEP]' + origin_data_list[2][i] + '[SEP]',
            origin_data_list[1][i] + '[SEP]' + origin_data_list[2][i],
            label=0 if origin_data_list[3][i] == 'neutral' else 1 if origin_data_list[3][i] == 'positive' else -1

        ))

    return instances_train


def adj_dependency_tree(arguments, max_length=80):
    nlp = spacy.load('en_core_web_sm')
    depend = []
    depend1 = []
    doc = nlp(str(arguments))
    d = {}
    i = 0
    for (_, token) in enumerate(doc):
        if str(token) in d.keys():
            continue
        d[str(token)] = i
        i = i + 1
    for token in doc:
        depend.append((token.text, token.head.text))
        depend1.append((d[str(token)], d[str(token.head)]))

    ze = np.identity(max_length)
    for (i, j) in depend1:
        if i >= max_length or j >= max_length:
            continue
        ze[i][j] = 1
    D = np.array(np.sum(ze, axis=1))
    D = np.matrix(np.diag(D))
    DSN = np.sqrt(D ** -1)
    DN = D ** -1
    return ze, DN, DSN


class ZXinDataset(dataset.Dataset):
    def __init__(self, instances):
        super(ZXinDataset, self).__init__()
        self.instances = instances

    def __getitem__(self, index):
        item = self.instances[index].x
        label = self.instances[index].label
        if label == 0:
            label = torch.tensor([0, 1, 0])
        elif label == 1:
            label = torch.tensor([0, 0, 1])
        else:
            label = torch.tensor([1, 0, 0])
        return item, label, self.instances[index].dt

    def __len__(self):
        return len(self.instances)
