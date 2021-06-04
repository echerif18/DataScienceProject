import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from Classifier import Classifier

def read_node_label(embeddings=None, label_file=None):
    fin = open(label_file, 'r')
    X = []
    Y = []
    label = {}

    for line in fin:
        a = line.strip('\n').split(' ')
        label[a[0]] = a[1]

    fin.close()
    for i in embeddings:
        X.append(i)
        Y.append(label[str(i)])

    return X, Y

def node_classification(embeddings, label_path, name, size):
    X, Y = read_node_label(embeddings, label_path, )

    f_c = open('%s_classification_%d.txt' % (name, size), 'w')

    all_ratio = []

    for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #print(" Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(dual=False,max_iter=10000), name=name)
        results = clf.split_train_evaluate(X, Y, tr_frac)
        avg = 'macro'
        f_c.write(name + ' train percentage: ' + str(tr_frac) + ' F1-' + avg + ' ' + str('%0.5f' % results[avg]))

        print(name ,' train percentage: ' , str(tr_frac) + ' F1-' + avg + ' ' , str('%0.5f' % results[avg]) )
        all_ratio.append(results[avg])
        f_c.write('\n')


def plot_embeddings(embeddings, label_file, name, method):
    X, Y = read_node_label(embeddings, label_file)

    emb_list = np.array([embeddings[k] for k in X])

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.legend()
    plt.title(method + " result of the dataset " + name)
    plt.savefig('%s.png' % (name))  # or '%s.pdf'%name
    plt.show()
