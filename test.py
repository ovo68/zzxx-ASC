import torch
from transformers import BertModel, BertTokenizer

bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

inputs = ['i am god', 'you are my son']


padded_sequence_ab = tokenizer(inputs, padding=True)
print("Padded sequence(A,B):", padded_sequence_ab["input_ids"])
print(type(padded_sequence_ab["input_ids"]))

print("Attention mask(A,B):", padded_sequence_ab["attention_mask"])
temp = torch.tensor(padded_sequence_ab["input_ids"])
print(temp.shape)
print(temp)

output = bert(temp)
print(output[0].shape)
print(output[0])
