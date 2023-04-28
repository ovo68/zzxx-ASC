import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, logging

logging.set_verbosity_error()


class EmbeddingLayer(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.bilstm = nn.LSTM(
        #     input_size=config.lstm_embedding_size,  # 1024
        #     hidden_size=config.hidden_size // 2,  # 1024
        #     batch_first=True,
        #     num_layers=2,
        #     dropout=config.lstm_dropout_prob,  # 0.5
        #     bidirectional=True
        # )
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            input_size=768,  #
            hidden_size=256,  # 双向的LSTM，256*2
            batch_first=True,
            num_layers=2,
            dropout=0.5,  # 0.5
            bidirectional=True
        )

    def forward(self, inputs):
        """
            单句级处理
        """
        # # 分词
        # tokens = self.tokenizer.tokenize(inputs)
        # # 得到每个词在bert中对应的id
        # tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        # # 将id转化为embedding
        # temp = torch.tensor(tokens_id)  # [x]
        # tokens_id_tensor = temp.unsqueeze(0)  # [x] => [x,x]
        # output = self.bert(tokens_id_tensor)
        # # print(type(output))
        # # print(output[0].shape)  # [x,sequence_length,768]
        # sequence_output = output[0]

        """
            多句级处理[[],[],...[]]
        """
        # padded_sequence_ab = self.tokenizer(inputs, padding=True)
        padded_sequence_ab = self.tokenizer(inputs, max_length=80, padding='max_length')

        tokens_id_tensor = torch.tensor(padded_sequence_ab["input_ids"])

        output = self.bert(tokens_id_tensor)
        sequence_output = output[0]

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        lstm_output, (h, c) = self.lstm(sequence_output)  # extract the 1st token's embeddings

        return lstm_output


if __name__ == '__main__':
    model = EmbeddingLayer()
    model("I charge it at night and skip taking the cord with me because of the good battery life.")
