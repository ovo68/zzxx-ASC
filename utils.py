import numpy as np
import spacy
import torch
from pandas import read_csv
from senticnet.senticnet import SenticNet
from torch import nn
from torch.nn import init
from torch.utils.data import dataset
from transformers import BertModel, BertTokenizer


class Instance:
    def __init__(self, no, sentence, aspect, polarity, dt, kt, at, x, label) -> None:
        super().__init__()
        self.target = None
        self.no = no
        self.sentence = sentence
        self.aspect = aspect
        self.polarity = polarity
        """
            x = ([CLS] s [SEP] a [SEP])
            x = (s [SEP] a)
        """
        self.x = x
        self.label = label

        # 句法依赖矩阵
        self.dt = dt
        # 常识知识矩阵
        self.kt = self.generate_kt()
        # 语义矩阵
        self.at = self.generate_at()
        # self.at = at

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

    def generate_at(self):

        bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        lstm = nn.LSTM(
            input_size=768,  #
            hidden_size=256,  # 双向的LSTM，256*2
            batch_first=True,
            num_layers=2,
            dropout=0.5,  # 0.5
            bidirectional=True
        )

        padded_sequence_ab = tokenizer(self.x, max_length=80, padding='max_length')

        tokens_id_tensor = torch.tensor(padded_sequence_ab["input_ids"])

        output = bert(torch.unsqueeze(tokens_id_tensor, dim=0))
        lstm_output, (h, c) = lstm(output[0])
        # print(lstm_output.shape)
        sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, d_out=80, h=2)
        sa_output = sa(lstm_output, lstm_output, lstm_output)
        return torch.squeeze(sa_output)

    # def generate_at(self):

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
    f.close()
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


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, d_out, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_out)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
