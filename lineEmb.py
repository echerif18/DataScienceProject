from torch.autograd import Variable
import torch.optim as optim
from LossNegSampling import *
import time, math
import networkx as nx
from sys import platform
from randomWalk import *
from nltk.util import ngrams

USE_CUDA = False

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

class lineEmb():

    def __init__(self, edge_file, name, social_edges=None, emb_size=128, epoch=4, batch_size=256, shuffel=True, neg_samples=5):
        self.emb_size = emb_size
        self.shuffel = shuffel
        self.neg_samples = neg_samples
        self.batch_size = batch_size
        self.epoch = epoch
        self.name = name
        self.learning_rate = 0.01 # init learning rate
        self.learning_rate_f = 0.001 # finalize learning rate
        self.G = nx.read_edgelist(edge_file)
        self.social_edges = social_edges
        self.index2word = dict()
        self.word2index = dict()
        self.build_vocab()
        self.lossList = []
        self.lossListEpochs=[] #to draw a loss-epoch graph

    def getBatch(self, batch_size, train_data):
        if self.shuffel == True:
            random.shuffle(train_data)

        sindex = 0
        eindex = batch_size
        while eindex < len(train_data):
            batch = train_data[sindex: eindex]
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield batch

        if eindex >= len(train_data):
            batch = train_data[sindex:]
            yield batch

    def prepare_node(self, node, word2index):
        return Variable(LongTensor([word2index[str(node)]]))

    def prepare_sequence(self, seq, word2index):
        idxs = list(map(lambda w: word2index[w], seq))
        return Variable(LongTensor(idxs))

    def build_vocab(self):
        self.social_nodes = []

        for u, v in self.social_edges:
            self.social_nodes.append(u)
            self.social_nodes.append(v)

        self.all_nodes = list(set(self.social_nodes))

        self.word2index = {}
        for vo in self.all_nodes:
            if self.word2index.get(vo) is None:
                self.word2index[str(vo)] = len(self.word2index)

        self.index2word = {v: k for k, v in self.word2index.items()}

    def prepare_trainData(self,walk_length,window_size,num_walks):
        print('prepare training data ...')
        w = randomWalk(self.G)
        walks = w.simulate_walks(num_walks, walk_length)
        flatten = lambda list: [item for sublist in list for item in sublist]
        windows = flatten([list(ngrams(c, window_size * 2+1)) for c in walks])
        train_data = []

        for window in windows:
            for i in range(window_size * 2 + 1):
                train_data.append((window[window_size], window[i]))

        u_p = []
        v_p = []

        for tr in train_data:
            u_p.append(self.prepare_node(tr[0], self.word2index).view(1, -1))
            v_p.append(self.prepare_node(tr[1], self.word2index).view(1, -1))

        train_samples = list(zip(u_p, v_p))
        print(len(train_samples), 'samples are ready ...')
        return train_samples

    def negative_sampling(self, targets, k):
        batch_size = targets.size(0)
        neg_samples = []

        for i in range(batch_size):
            nsample = []
            target_index = targets[i].data.cpu().tolist()[0] if USE_CUDA else targets[i].data.tolist()[0]
            v_node = self.index2word[target_index]

            while len(nsample) < k:  # num of negative samples
                neg = random.choice(self.all_nodes)
                if (neg != v_node):
                    nsample.append(neg)
                else:
                    continue
            neg_samples.append(self.prepare_sequence(nsample, self.word2index).view(1, -1))

        return torch.cat(neg_samples)

    def train(self, k_com, num_walks, walk_length, window):
        final_losses = []
        train_data = self.prepare_trainData(walk_length, window, num_walks)
        model = LossNegSampling(len(set(self.all_nodes)), self.emb_size, k_com, len(train_data), self.epoch)
        if USE_CUDA:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        #define node_labels
        node_labels = {}
        for epoch in range(self.epoch):
                gammaList = [] # list to hold gamma values
                learningList = [] # list to hold learning rates
                t1 = time.time()
                for i, batch in enumerate(self.getBatch(self.batch_size, train_data)):
                    inputs, targets = zip(*batch)

                    inputs = torch.cat(inputs)  # B x 1
                    targets = torch.cat(targets)  # B x 1
                    I=np.unique(inputs.data.cpu().numpy())
                    model.t = model.t + len(I)
                    negs = self.negative_sampling(targets, self.neg_samples)

                    gammaValue, final_loss, cluster_choice = model(inputs, targets, negs)
                    gammaList.append(gammaValue)
                    cluster_labels = cluster_choice.data.cpu().numpy() #set node labels, we can visualize them or write them anywhere.

                    # add node label with index.
                    for i in range(len(inputs)):
                        id = inputs[i].data.cpu().numpy()[0]
                        node_idx = self.index2word[id]
                        node_labels[node_idx] = cluster_labels[i]

                    # calculate learning rate
                    lr = self.learning_rate - ((self.learning_rate - self.learning_rate_f) * model.t) / (self.epoch * len(train_data))
                    learningList.append(lr)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                    # do backpropagation
                    model.zero_grad()
                    final_loss.backward(retain_graph=True)
                    optimizer.step()

                    final_losses.append(final_loss.data.cpu().numpy())
                    if(self.epoch == 1):
                        self.lossList.append(np.mean(final_losses))

                self.lossListEpochs.append(np.mean(final_losses))
                t2 = time.time()
                print(self.name, 'Loss: %0.3f ' % np.mean(final_losses), 'Epoch time: ', '%0.4f' % (t2 - t1),
                      'Dimension size:', self.emb_size,'Gamma: ', np.mean(gammaList),'Learning rate: ',np.mean(learningList))

        final_emb = {}
        normal_emb = {}

        #check current operating system and create a file
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            labelFile = open("datasets/" + self.name + "/" + self.name + "-node_labels.txt", 'w')
        elif platform == "win32":
            labelFile = open("datasets\\" + self.name + "\\" + self.name + "-node_labels.txt", 'w')

        node_number_list = list(node_labels.keys())
        node_label_list = list(node_labels.values())

        for index in range(0, len(node_number_list)):
            labelFile.write(str(node_number_list[index]) + " " + str(node_label_list[index]) + "\n")
        labelFile.close()

        for w in self.all_nodes:
            normal_emb[w] = model.get_emb(self.prepare_node(w, self.word2index))
            vec = [float(i) for i in normal_emb[w].data.cpu().numpy()[0]]
            final_emb[int(w)] = vec

        return final_emb