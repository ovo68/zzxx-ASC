import torch
from senticnet.senticnet import SenticNet

from utils import load_data


def testSN():
    sn = SenticNet()  # 可以指定语言参数，不写默认英语
    concept_info = sn.concept('love')
    polarity_value = sn.polarity_value('love')
    polarity_label = sn.polarity_label('love')
    moodtags = sn.moodtags('love')
    semantics = sn.semantics('love')
    sentics = sn.sentics('love')
    print('concept_info:', concept_info)
    print('polarity_value:', polarity_value)
    print('polarity_label:', polarity_label)
    print('moodtags:', moodtags)
    print('semantics:', semantics)
    print('sentics:', sentics)


# sentence = 'I charge it at night and skip taking the cord with me because of the good battery life.'

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# token = tokenizer.tokenize(sentence)
# print(type(token))
# print(token)

# concept_info = sn.concept('cord')
# polarity_value = sn.polarity_value('cord')
# polarity_label = sn.polarity_label('cord')
# moodtags = sn.moodtags('cord')
# semantics = sn.semantics('cord')
# sentics = sn.sentics('cord')
#


def generate_1(instance):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    #
    # token = tokenizer.tokenize(sentence)
    token = instance.sentence.split(' ')
    common_matrix = torch.eye(80, 80)
    for index, word in enumerate(token):
        common_word = generate_single_word(word, instance.polarity)
        if len(common_word) == 0:
            continue
        print('common_word len', len(common_word))
        for i in range(len(common_word)):
            if (common_matrix[index, len(token) + i].item() == float(0)) & (
                    common_matrix[len(token) + i, index].item() == float(0)):

                common_matrix[index, len(token) + i] = torch.tensor(1)
                common_matrix[len(token) + i, index] = torch.tensor(1)

                print(index, len(token) + i)
                print(len(token) + i, index)
            else:
                continue
        print(word, index, '=>', common_word)

    n = 0
    for i in common_matrix.reshape(-1):
        element = i.item()
        if element == float(1):
            n = n + 1
        print('%.2f' % element, end=' ')
    print('\n')
    print(n)
    print(common_matrix)


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


def generate_common_words(instances):
    common_words = []
    for instance in instances:
        common_word = []
        sn = SenticNet()
        wordlist = instance.sentence.split(' ')

        for word in wordlist:

            # 防止有的词不存在于senticnet词典，导致程序报错！
            try:
                concept_info = sn.concept(word)
                polarity_label = concept_info['polarity_label']
                polarity_value = concept_info['polarity_value']

                if instance.polarity == polarity_label:
                    moodtags = concept_info['moodtags']
                    semantics = concept_info['semantics']
                    common_word.extend(semantics)
                    common_word.extend(moodtags)
                elif instance.polarity == polarity_label:
                    moodtags = concept_info['moodtags']
                    semantics = concept_info['semantics']
                    common_word.extend(semantics)
                    common_word.extend(moodtags)
                else:
                    continue


            except:
                continue

        # print("before:", len(common_words))
        common_word = list(set(common_word))
        # print("after:", len(common_words))
        common_words.append(common_word)
    return common_words


#
# w = generate_common_words(load_data())
# print(len(w))
# print(w)


if __name__ == '__main__':
    generate_1(load_data()[1])
    # testSN()
