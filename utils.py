import numpy as np
import spacy
import torch
from pandas import read_csv
from senticnet.senticnet import SenticNet
from torch.utils.data import dataset


class Instance:
    def __init__(self, no, sentence, aspect, polarity, dt, kt, at, x, label) -> None:
        super().__init__()
        self.target = None
        self.no = no
        self.sentence = sentence
        self.aspect = aspect
        self.polarity = polarity
        # 句法依赖矩阵
        self.dt = dt
        # 常识知识矩阵
        self.kt = self.generate_kt()
        # 语义矩阵
        self.at = at

        """
            x = ([CLS] s [SEP] a [SEP])
            x = (s [SEP] a)
        """
        self.x = x
        self.label = label

    def generate_kt(self):
        token = self.sentence.split(' ')
        common_matrix = torch.eye(80, 80)
        for index, word in enumerate(token):
            common_word = generate_single_word(word, self.polarity)
            if len(common_word) == 0:
                continue

            for i in range(len(common_word)):
                if (common_matrix[index, len(token) + i].item() == float(0)) & (
                        common_matrix[len(token) + i, index].item() == float(0)):

                    common_matrix[index, len(token) + i] = torch.tensor(1)
                    common_matrix[len(token) + i, index] = torch.tensor(1)

                else:
                    continue
        return common_matrix

    def __str__(self) -> str:
        return 'no:' + str(
            self.no) + ',sentence=' + self.sentence + \
            ',aspect=' + self.aspect + ',polarity=' + self.polarity + ',x=' + self.x


def load_data(path='demo.csv'):
    f = open(path, encoding='UTF-8')
    column_names = ['id', 'Sentence', 'Aspect_Term', 'polarity', 'from', 'to']
    origin_data = read_csv(f, names=column_names, header=None, skiprows=1)

    instances_train = []
    origin_data_list = [origin_data[column_names[i]].tolist() for i in range(len(column_names))]
    for i in range(len(origin_data_list[0])):
        _, dt, _ = adj_dependency_tree(origin_data_list[1][i])

        # TODO 常识知识矩阵构建
        kt = torch.randn((80, 80), dtype=torch.float32)

        # TODO 语义矩阵构建
        at = torch.randn((80, 80), dtype=torch.float32)

        instances_train.append(Instance(
            origin_data_list[0][i],
            origin_data_list[1][i],
            origin_data_list[2][i],
            origin_data_list[3][i],
            torch.tensor(dt, dtype=torch.float32),
            kt,
            at,
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


def generate_single_word(word, polarity):
    sn = SenticNet()
    common_word = []
    try:
        concept_info = sn.concept(word)
        polarity_label = concept_info['polarity_label']
        polarity_value = concept_info['polarity_value']

        if polarity == polarity_label:
            moodtags = concept_info['moodtags']
            semantics = concept_info['semantics']
            common_word.extend(semantics)
            common_word.extend(moodtags)
        elif polarity == polarity_label:
            moodtags = concept_info['moodtags']
            semantics = concept_info['semantics']
            common_word.extend(semantics)
            common_word.extend(moodtags)
        else:
            return common_word


    except:
        return []

        # print("before:", len(common_words))
    common_word = list(set(common_word))
    return common_word


def normalize(adj):
    m = np.mean(adj)
    mx = max(adj)
    mn = min(adj)
    return [(float(i) - m) / (mx - mn) for i in adj]


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
        return item, label, self.instances[index].dt, self.instances[index].kt, self.instances[index].at

    def __len__(self):
        return len(self.instances)
