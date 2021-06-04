import torch
import torch.nn as nn
import numpy as np
import math
class LossNegSampling(nn.Module):

    def __init__(self, num_nodes, emb_dim, k_com, tdSize, noEpoch):
        super(LossNegSampling, self).__init__()
        self.embedding_u = nn.Embedding(num_nodes, emb_dim)  # embedding  u
        self.embedding_v = nn.Embedding(num_nodes, emb_dim)  # embedding  V
        self.logsigmoid = nn.LogSigmoid()
        self.embedding_com = nn.Embedding(k_com, emb_dim)
        initrange = (2.0 / (num_nodes + emb_dim)) ** 0.5  # Xavier init 2.0/sqrt(num_nodes+emb_dim)
        self.embedding_u.weight.data.uniform_(-initrange, initrange)  # initialization of the u with uniform distribution
        self.embedding_v.weight.data.uniform_(-0.0, 0.0)  # v init
        self.k_com = k_com  # number of communities(clusters in the dataset)
        self.gamma_0 = 0.01 # initialize gamma
        self.t = 0 # initialize t counter
        self.trainDataSize = tdSize
        self.epochNumber = noEpoch
        inits = self.embedding_u(torch.LongTensor(np.random.randint(0, num_nodes - 1, size=k_com)))
        self.embedding_com.weight.data.copy_(inits)

    def forward(self, u_node, v_node, negative_nodes):
        u_embed = self.embedding_u(u_node)  # n x 1 x d  edge (u,v)
        v_embed = self.embedding_v(v_node)  # n x 1 x d
        negs = -self.embedding_v(negative_nodes)  # n x K x d  neg samples

        positive_score = v_embed.bmm(u_embed.transpose(1, 2)).squeeze(2)  # nx1
        negative_score = torch.sum(negs.bmm(u_embed.transpose(1, 2)).squeeze(2), 1).view(negative_nodes.size(0),-1)  # nxK -> nx1
        sum_all = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
        emb_loss = -torch.mean(sum_all) #node embedding cost

        n = u_embed.shape[0]
        k = self.k_com

        z = u_embed.repeat(1, k, 1)
        mu = self.embedding_com.weight.repeat(n, 1, 1)

        dist = (z - mu).norm(2, dim=2).reshape((n, k))
        # add argmin
        cluster_choice = torch.argmin(dist, dim=1)
        k_loss = (dist.min(dim=1)[0] ** 2).mean() #clustering cost

        #gamma calculation
        #self.t = self.t + u_embed.size()[0]
        gammaValue = self.gamma_0 * (10 ** ((-self.t * math.log10(self.gamma_0)) / (self.epochNumber * self.trainDataSize)))

        final_loss = emb_loss + gammaValue*k_loss #total loss function
        return [gammaValue, final_loss, cluster_choice] #return gamma, final loss and cluster choices

    def get_emb(self, input_node):
        embeds = self.embedding_u(input_node)  ### u
        return embeds
