import torch
from torch.nn.parameter import Parameter

weight = Parameter(torch.FloatTensor(2,6, 8))
print(weight.shape)
print(weight)

# bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#
# inputs = ['i am god', 'you are my son']
#
# padded_sequence_ab = tokenizer(inputs, padding=True)
#
# temp = torch.tensor(padded_sequence_ab["input_ids"])
# print('input_ids shape:', temp.shape)
#
# output = bert(temp)
# print('bert shape:', output[0].shape)
# lstm = nn.LSTM(
#     input_size=768,  #
#     hidden_size=256,  # 双向的LSTM，256*2
#     batch_first=True,
#     num_layers=2,
#     dropout=0.5,  # 0.5
#     bidirectional=True
# )
#
# lstm_output, (_, _) = lstm(output[0])
# print('lstm_output shape:', lstm_output.shape)
# adj = [
#     [
#         [0.5, 0., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0., 0.],
#         [0., 0., 0.5, 0., 0., 0.],
#         [-0., -0., -0., 1., -0., -0.],
#         [0., 0., 0., 0., 1., 0.],
#         [0., 0., 0., 0., 0., 1.]
#     ],
#     [
#         [0.5, 0., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0., 0.],
#         [0., 0., 0.5, 0., 0., 0.],
#         [-0., -0., -0., 0.5, -0., -0.],
#         [0., 0., 0., 0., 1., 0.],
#         [0., 0., 0., 0., 0., 1.]
#     ]
# ]
# adj_tensor = torch.tensor(adj)
# print('adj_tensor shape:', adj_tensor.shape)
# t = torch.bmm(adj_tensor, lstm_output)
# print(t.shape)
#
# weights = torch.randn(2, 512, 3)
# support = torch.matmul(lstm_output, weights)
# gcn_out = torch.matmul(adj_tensor, support)
#
# print("gcn_out shape:", gcn_out.shape)
# print("--------------------------------------")

#
# # inputs = torch.randn(2, 6, 512)
# inputs = torch.randn(2, 6, 8)
# # print("mat1=", mat1)
#
# # weights = torch.randn(2, 512, 1024)
# weights = torch.randn(2, 8, 3)
# # print("mat2=", mat2)
#
# support = torch.matmul(inputs, weights)
# print("support=", support, support.shape)
