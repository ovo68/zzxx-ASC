import sys
import numpy as np
from my_senticnet import senticnet
from utils import normalize
import pickle
base_path = sys.path[0] + '/../data/'


def onehot(x, primary_mood):
    one_hot = np.zeros(len(primary_mood), dtype='float32')
    one_hot[primary_mood[x]] = 1
    return one_hot


def node2vec():
    filename = base_path + "knowledge/affectivespace/affectivespace.csv"
    fr = open(filename)
    word_vec = {}
    for line in fr.readlines():
        tokens = line.rstrip().split(",")
        word_vec[tokens[0]] = np.asarray(tokens[-100:], dtype='float32')
    return word_vec


class Graph():
    def __init__(self):
        self.graph = senticnet
        self.primary_mood_to_id = self.extract_primary_mood()
        self.secondary_mood_to_id = self.extract_secondary_mood()
        self.mood_to_id = self.extract_mood()
        self.polarity_label_to_id = self.extract_polarity_label()
        self.word2vec = node2vec()
        self.words = {}

    def extract_primary_mood(self):
        primary_mood = {}
        num = 0
        for word in self.graph.keys():
            tmp = self.graph[word][4]
            if tmp not in primary_mood:
                primary_mood[tmp] = num
                num += 1
        return primary_mood

    def extract_secondary_mood(self):
        secondary_mood = {}
        num = 0
        for word in self.graph.keys():
            tmp = self.graph[word][5]
            if tmp not in secondary_mood:
                secondary_mood[tmp] = num
                num += 1
        return secondary_mood

    def extract_mood(self):
        mood = {}
        num = 0
        for word in self.graph.keys():
            tmp = self.graph[word][4]
            if tmp not in mood:
                mood[tmp] = num
                num += 1
            tmp = self.graph[word][5]
            if tmp not in mood:
                mood[tmp] = num
                num += 1
        return mood

    def extract_polarity_label(self):
        polarity_label = {}
        num = 0
        for word in self.graph.keys():
            tmp = self.graph[word][6]
            if tmp not in polarity_label:
                polarity_label[tmp] = num
                num += 1
        return polarity_label

    def get_vec(self, word):
        if word not in self.word2vec:
            return np.zeros(100, dtype='float32')
        else:
            return self.word2vec[word]

    def infoextract(self, word):
        vec = self.get_vec(word)
        if word not in self.graph:
            return np.zeros(5 + len(self.mood_to_id) * 2 + len(self.polarity_label_to_id), dtype='float32'), vec, []
        data = self.graph[word]
        ans = []
        pleasantness_value = float(data[0])
        ans.append(pleasantness_value)
        attention_value = float(data[1])
        ans.append(attention_value)
        sensitivity_value = float(data[2])
        ans.append(sensitivity_value)
        aptitude_value = float(data[3])
        ans.append(aptitude_value)
        primary_mood = data[4]
        primary_mood_onehot = onehot(primary_mood, self.mood_to_id)
        ans += list(primary_mood_onehot)
        secondary_mood = data[5]
        secondary_mood_onehot = onehot(secondary_mood, self.mood_to_id)
        ans += list(secondary_mood_onehot)
        polarity_label = data[6]
        polarity_label_onehot = onehot(polarity_label, self.polarity_label_to_id)
        ans += list(polarity_label_onehot)
        polarity_value = float(data[7])
        ans.append(polarity_value)

        # ans += list(vec)
        ans = np.array(ans, dtype='float32')
        semantics = []
        semantics1 = data[8]
        semantics.append(semantics1)
        semantics2 = data[9]
        semantics.append(semantics2)
        semantics3 = data[10]
        semantics.append(semantics3)
        semantics4 = data[11]
        semantics.append(semantics4)
        semantics5 = data[12]
        semantics.append(semantics5)
        return ans, vec, semantics


def build_graph(graph, text, max_sequence_len=120, max_node_num=360):
    adj = np.zeros((max_sequence_len + max_node_num, max_sequence_len + max_node_num))
    nodes_id = {}
    num = max_sequence_len
    words = text.split()
    features = np.zeros((max_sequence_len+ max_node_num, 123), dtype='float32')
    add_node = []
    for i in range(len(words)):
        feature, vec, semantics = graph.infoextract(words[i])
        print("words[i], semantics****: ", words[i], semantics)
        # print(feature.shape, vec.shape)
        features[i] = np.concatenate((feature, vec), axis=0)
        for node in semantics:
            if node not in nodes_id:
                add_node.append(node)
                nodes_id[node] = num
                feature_node, vec_node, semantics_node = graph.infoextract(node)
                features[nodes_id[node]]=np.concatenate((feature_node, vec_node))
                for node_2 in semantics_node:
                    if node_2 in nodes_id:
                        print("node, node_2****: ", node, node_2)
                        adj[nodes_id[node], nodes_id[node_2]] = 1
                num += 1
            adj[i, nodes_id[node]] = 1
            if len(nodes_id) >= max_node_num:
                break
        if len(nodes_id) >= max_node_num:
            break

    for new_node in add_node:
        feature, vec, semantics = graph.infoextract(new_node)
        print("new_node, semantics----: ", new_node, semantics)
        for node in semantics:
            if node not in nodes_id:
                # add_node.append(node)
                nodes_id[node] = num
                feature_node, vec_node, semantics_node = graph.infoextract(node)
                features[nodes_id[node]] = np.concatenate((feature_node, vec_node))
                for node_2 in semantics_node:
                    if node_2 in nodes_id:
                        print("node node_2----: ", node, node_2)
                        adj[nodes_id[node], nodes_id[node_2]] = 1
                num += 1
            adj[nodes_id[new_node], nodes_id[node]] = 1
            if len(nodes_id) >= max_node_num:
                break
        if len(nodes_id) >= max_node_num:
            break
    adj += np.eye(adj.shape[0], adj.shape[0])
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj)
    adj = np.array(adj, dtype='float32')
    features = np.array(features, dtype='float32')
    return adj, features


def build_graph_file(fname, file_out, graph):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    all_data = {}
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        # polarity = lines[i + 2].strip()
        text = text_left + " " + aspect + " " + text_right
        adj, features = build_graph(graph, text)
        all_data[int(i/3)] = [adj, features]
    pickle.dump(all_data, open(file_out, 'wb'))


def run(dataset, graph):
    dataset_files = {
        'train': base_path + 'tmp/' + dataset + '_train.txt',
        'test': base_path + 'tmp/' + dataset + '_test.txt',
        'train_to': base_path + 'store/' + dataset + '_train_knowledge_graph.dat',
        'test_to': base_path + 'store/' + dataset + '_test_knowledge_graph.dat',
    }
    build_graph_file(dataset_files['train'], dataset_files['train_to'], graph)
    build_graph_file(dataset_files['test'], dataset_files['test_to'], graph)

if __name__ == '__main__':
    graph = Graph()
    # sentence = "i have experienced no problems , works as anticipated ."
    adj, features = build_graph(graph, text='great laptop that offers many great features !')
    print(adj.shape, adj)
    print(features.shape, features)
    # run(dataset='laptop14', graph=graph)
    # run(dataset='twitter', graph=graph)
    # run(dataset='restaurants14', graph=graph)
    # run(dataset='restaurants15', graph=graph)
    # run(dataset='restaurants16', graph=graph)
