from tkinter import messagebox
from lineEmb import *
from matplotlib import pyplot as plt
from tkinter import *
from sys import platform
import util_functions as util
from gensim.models import Word2Vec
import graph
import warnings
warnings.filterwarnings("ignore")

global selectedSet, edge_file, label_file, groundtruth_file, k_com, mainWindow, shuffleStatus, epoch_default, cycle_default, get_embedding_size, get_batch_size, windowSize, walkLength

def drawLoss(model):
    plt.figure()
    if(model.epoch == 1):
        plt.plot(model.lossList, label="Total Loss")
        plt.xlabel("Number of Epochs: 1")
    else:
        plt.plot(model.lossListEpochs, label="Total Loss")
        plt.xlabel("Number of Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.title("Total loss of the model")
    plt.show()

def definePaths(dataset):
    # define global variables
    global edge_file
    global label_file
    global groundtruth_file
    global k_com

    # define dictionary
    kcom_dic = {'airport': 5, 'citeseer': 6, 'cora': 7, 'facebook': 10, 'wiki': 19}

    #assign label and edge files appropriately for different operating systems.
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        #if running operating system is linux or mac then
        edge_file = "datasets/" + dataset + "/" + dataset + "-edgelist.txt"
        groundtruth_file = "datasets/" + dataset + "/" + dataset + "-label.txt"  # original label file
        label_file = "datasets/" + dataset + "/" + dataset + "-node_labels.txt"  # evaluated label file
    elif platform == "win32":
        # if running operating system is windows then
        edge_file = "datasets\\"+dataset+"\\"+dataset+"-edgelist.txt"
        groundtruth_file = "datasets\\"+dataset+"\\"+dataset+"-label.txt" # original label file
        label_file = "datasets\\" + dataset + "\\" + dataset + "-node_labels.txt" # evaluated label file
    k_com = kcom_dic.get(dataset)

def startProcess(dataset, embeding_size_status, batch_size_status, epoch_status, cycle_status, shuffle_status, walk_length, window_size):
    definePaths(dataset)
    opened_edge_file = open(edge_file, 'r')
    social_edges = []
    shuffle_data = False

    for line in opened_edge_file:
        holder = line.strip('\n').split(' ')
        social_edges.append((holder[0], holder[1]))

    # set shuffle true or false
    if shuffle_status == 1:
        shuffle_data = True

    model = lineEmb(edge_file, dataset, social_edges, embeding_size_status, epoch_status, batch_size_status, shuffle_data)
    embeddings_gemsec = model.train(k_com,cycle_status,walk_length,window_size)

    # draw total loss graphic
    drawLoss(model)

    # node classification
    print("Gemsec F1 Scores:")
    util.node_classification(embeddings_gemsec, groundtruth_file, dataset, size=embeding_size_status)
    util.plot_embeddings(embeddings_gemsec, groundtruth_file, dataset, "Gemsec") # visualization_groundtruth_labels

    # Deepwalk
    print("Deepwalk F1 Scores:")
    G = graph.load_edgelist(edge_file, undirected=True)
    t3 = time.time()
    walks = graph.build_deepwalk_corpus(G, num_paths=cycle_status, path_length=walk_length, alpha=0, rand=random.Random(0))
    nodes = G.number_of_nodes()
    print('embeddings for ==>', dataset)
    embeddings_deepwalk = deepWalk(walks, dataset, embeding_size_status, nodes, window_size)
    t4 = time.time()
    util.node_classification(embeddings_deepwalk, groundtruth_file, dataset, size=embeding_size_status)
    util.plot_embeddings(embeddings_deepwalk, groundtruth_file, dataset, "Deepwalk")
    print("Duration: " + str(t4 - t3))

# deep walk func
def deepWalk(walks=None, name='Facebook', emb_size=128, nodes=None, window_size=5):
    print("Training...")

    output = "./emb/%s.deepwalk" % (name)

    t1 = time.time()

    model = Word2Vec(walks, size=emb_size, window=window_size, min_count=0, workers=2, hs=1, sg=1)

    t2 = time.time()
    print(t2 - t1)

    model.wv.save_word2vec_format('%s.emb' % name)

    f = open('%s.emb' % name, 'r')
    emb = {}
    for line in f:
        a = line.strip('\n').split(" ")
        emb[int(a[0])] = [float(i) for i in a[1:]]

    final_emb = {}
    for i in range(nodes):
        final_emb[i] = emb[i]

    return final_emb

def informUser(dataset):
    messagebox.showinfo("Information", "Analyzing '"+dataset+"' dataset.")
    close_window()

def close_window():
    global mainWindow
    mainWindow.destroy()

def taskSelection():
    setList = ["airport","citeseer", "cora", "facebook", "wiki"]
    selection = selectedSet.get()
    shuffle_status = shuffleStatus.get()
    epoch_status = epoch_default.get()
    cycle_status = cycle_default.get()
    dataSet = setList[selection]
    window_size = windowSize.get()
    walk_length = walkLength.get()
    embeding_size_status = get_embedding_size.get()
    batch_size_status = get_batch_size.get()
    informUser(dataSet)
    startProcess(dataSet, embeding_size_status, batch_size_status, epoch_status, cycle_status, shuffle_status, walk_length, window_size)

def defineDataset():
    global selectedSet, mainWindow, shuffleStatus, epoch_default, cycle_default, get_batch_size, get_embedding_size, windowSize, walkLength

    # list of offered embedding sizes
    embeddings_list = [
        32,
        64,
        128,
        256,
        512,
        1024
    ]
    # list of offered batch sizes
    batch_list = [
        32,
        64,
        128,
        256,
        512,
        1024
    ]

    # define main window
    mainWindow = Tk()
    mainWindow.title("Preferences:")
    mainWindow.geometry('850x150+300+250')
    mainWindow.resizable(0, 0)
    selectedSet = IntVar()
    # set default value for shuffle
    shuffleStatus = IntVar()
    shuffleStatus.set(1)
    # set default value for embedding size
    get_embedding_size = IntVar()
    get_embedding_size.set(embeddings_list[2])
    # set default value for batch size
    get_batch_size = IntVar()
    get_batch_size.set(batch_list[3])
    # set default value for number of epochs
    epoch_default = IntVar()
    epoch_default.set(8)
    # set default value for number of walks (n)
    cycle_default = IntVar()
    cycle_default.set(5)
    # set default value for window size
    windowSize = IntVar()
    windowSize.set(5)
    # set default value for walk length
    walkLength = IntVar()
    walkLength.set(40)

    Label(mainWindow, text="Select a dataset to analyze:").place(x=5, y=5)
    Radiobutton(mainWindow, text="Airport", variable=selectedSet, value=0).place(x=5, y=30)
    Radiobutton(mainWindow, text="Citeseer", variable=selectedSet, value=1).place(x=5,y=50)
    Radiobutton(mainWindow, text="Cora", variable=selectedSet, value=2).place(x=5,y=70)
    Radiobutton(mainWindow, text="Facebook", variable=selectedSet, value=3).place(x=5,y=90)
    Radiobutton(mainWindow, text="Wiki", variable=selectedSet, value=4).place(x=5,y=110)
    Label(mainWindow, text="Epoch Number:").place(x=200,y=5)
    Spinbox(mainWindow, from_=1, to=10, textvariable=epoch_default).place(x=200,y=30)
    Label(mainWindow, text="Number of Walks:").place(x=200,y=60)
    Spinbox(mainWindow, from_=1, to=10, textvariable=cycle_default).place(x=200,y=80)
    Label(mainWindow, text="Window Size:").place(x=400, y=5)
    Spinbox(mainWindow, from_=1, to=20, textvariable=windowSize).place(x=400, y=30)
    Label(mainWindow, text="Walk Length:").place(x=400, y=60)
    Spinbox(mainWindow, from_=1, to=80, textvariable=walkLength).place(x=400, y=80)
    Label(mainWindow, text="Embedding Size:").place(x=600,y=5)
    OptionMenu(mainWindow, get_embedding_size, *embeddings_list).place(x=600,y=30)
    Label(mainWindow, text="Batch Size:").place(x=600,y=60)
    OptionMenu(mainWindow, get_batch_size, *batch_list).place(x=600,y=80)
    Label(mainWindow, text="Shuffle:").place(x=750, y=5)
    Radiobutton(mainWindow, text="True", variable=shuffleStatus, value=1).place(x=750,y=30)
    Radiobutton(mainWindow, text="False", variable=shuffleStatus, value=0).place(x=750,y=50)
    Button(mainWindow, text="Execute", command=taskSelection).place(x=750,y=80)
    # show main window
    mainWindow.mainloop()

if __name__ == '__main__':
    defineDataset()







