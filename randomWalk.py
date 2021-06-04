import random

class randomWalk:

    def __init__(self, graph):
        self.G = graph

    def random_walk(self,walk_length, start_node):
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))

            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self,num_walks, walk_length):

        nodes= list(self.G.nodes())
        walks = []
        for walk_iter in range(num_walks):

            print(str(walk_iter+1), '/', str(num_walks))

            random.shuffle(nodes)

            for v in nodes:
                walks.append(self.random_walk(walk_length=walk_length, start_node=v))

        return walks
