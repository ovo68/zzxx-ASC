from pandas import read_csv
from torch.utils.data import dataset
from torch.utils.data import dataloader

from model.EmbeddingLayer import EmbeddingLayer


class Instance:
    def __init__(self, no, sentence, aspect, polarity, x, label) -> None:
        super().__init__()
        self.target = None
        self.no = no
        self.sentence = sentence
        self.aspect = aspect
        self.polarity = polarity
        """
            x = ([CLS] s [SEP] a [SEP])
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
        instances_train.append(Instance(
            origin_data_list[0][i],
            origin_data_list[1][i],
            origin_data_list[2][i],
            origin_data_list[3][i],
            '[CLS]' + origin_data_list[1][i] + '[SEP]' + origin_data_list[2][i] + '[SEP]',
            label=0 if origin_data_list[3][i] == 'neutral' else 1 if origin_data_list[3][i] == 'positive' else -1

        ))

    return instances_train


class ZXinDataset(dataset.Dataset):
    def __init__(self, instances):
        super(ZXinDataset, self).__init__()
        self.instances = instances

    def __getitem__(self, index):
        item = self.instances[index].x
        label = self.instances[index].label

        return item, label

    def __len__(self):
        return len(self.instances)


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

    for item, label in train_loader:
        lstm_output = net(item)
        print(lstm_output.shape)
