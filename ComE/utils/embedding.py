__author__ = 'ando'
import itertools
import logging as log
import math
import numpy as np
from scipy.special import expit as sigmoid

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.INFO)

#sdg gradient update
def gradient_update(positive_node_embedding, negative_nodes_embedding, neg_labels, _alpha):
    '''
      Perform stochastic gradient descent of the first and second order embedding.
      NOTE: using the cython implementation (fast_community_sdg_X) is much more fast
    '''
    fb = sigmoid(np.dot(positive_node_embedding, negative_nodes_embedding.T))  #  propagate hidden -> output
    gb = (neg_labels - fb) * _alpha# vector of error gradients multiplied by the learning rate
    return gb


def community_sdg(node_embedding, centroid, inv_covariance_mat, pi, k, _alpha, _lambda2, index, covariance_mat):
    '''
      Perform stochastic gradient descent of the community embedding.
      NOTE: using the cython implementation (fast_community_sdg_X) is much more fast
    '''
    grad = np.zeros(node_embedding[index].shape, dtype=np.float32)
    # node_loss =  0
    for com in range(k):
        diff = (node_embedding[index] - centroid[com])
        m = pi[index, com] * inv_covariance_mat[com]
        grad += np.dot(m, diff) * _lambda2

    return - np.clip((grad), -0.1 * _alpha, 0.1 * _alpha)
        # node_loss += pi[index, com] * np.log(multivariate_normal.pdf(node_embedding[index], centroid[com], covariance_mat[com]))
    # return - np.clip((grad), -0.1*_alpha, 0.1*_alpha), node_loss


def chunkize_serial(iterable, chunksize, as_numpy=False):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).

    >>> print(list(grouper(range(10), 3)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """

    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[np.array(doc) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        # memory opt: wrap the chunk and then pop(), to avoid leaving behind a dangling reference
        yield wrapped_chunk.pop()

def prepare_sentences(model, paths):
    '''
    :param model: current model containing the vocabulary and the index
    :param paths: list of the random walks. we have to translate the node to the appropriate index and apply the dropout
    :return: generator of the paths according to the dropout probability and the correct index
    '''
    for path in paths:
        # avoid calling random_sample() where prob >= 1, to speed things up a little:
        sampled = [model.vocab[node] for node in path
                   if node in model.vocab and (model.vocab[node].sample_probability >= 1.0 or model.vocab[node].sample_probability >= np.random.random_sample())]
        yield sampled

def batch_generator(iterable, batch_size=1):
    '''
    same as chunkize_serial, but without the usage of an infinite while
    :param iterable: list that we want to convert in batches
    :param batch_size: batch size
    '''
    args = [iterable] * batch_size
    return itertools.zip_longest(*args, fillvalue=None)

class RepeatCorpusNTimes():
    def __init__(self, corpus, n):
        '''
        Class used to repeat n-times the same corpus of paths
        :param corpus: list of paths that we want to repeat
        :param n: number of times we want to repeat our corpus
        '''
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            for document in self.corpus:
                yield document



class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


def xavier_normal(size, as_type=np.float32, gain=1):
    assert len(size) == 2
    std = gain * math.sqrt(2.0 / sum(size))
    return np.random.normal(size=size, loc=0, scale=std).astype(as_type)