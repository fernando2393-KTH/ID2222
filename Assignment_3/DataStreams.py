import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
import random
import copy
import math
from tqdm import tqdm


def union(cnt_1, cnt_2):
    new_cnt_1 = copy.deepcopy(cnt_1)
    for i in range(len(cnt_1.max_r)):
        new_cnt_1.max_r[i] = max(cnt_1.max_r[i], cnt_2.max_r[i])

    return new_cnt_1


def estimate_centrality(harmonic, cnt, cnt_old, t):
    for node in cnt.keys():
        harmonic[node] += (cnt[node].computeSize() - cnt_old[node].computeSize()) / t

    return harmonic


def computeHarmonic(graph):
    harmonic = {node: 0 for node in graph.nodes}
    for x in nx.nodes(graph):
        for y in nx.nodes(graph):
            if x != y:
                try:
                    shortest_path = nx.shortest_path(graph, y, x)
                    shortest_path = len(shortest_path) - 1
                    harmonic[x] += 1 / shortest_path
                except:
                    # Never ever do this
                    pass

    return harmonic


def rmse(dict1, dict2):
    error = 0
    for k, v in dict1.items():
        error += pow(v - dict2[k], 2)

    return pow((error / len(dict1)), 0.5)


def hyperball(graph, bits):
    print("Graph reverted")
    nodes = [node for node in graph.nodes]
    edges = [(v, w) for (v, w) in graph.edges]
    cnt = {node: FlajoletMartin(bits=bits) for node in nodes}  # Initialize a fjm counter for each node
    for node in tqdm(nodes):
        cnt[node].addElem(node)

    t = 1  # Threshold
    harmonic = {node: 0 for node in nodes}
    print("Starting value comparisons...")
    while True:  # While value changes
        change = False  # False --> no value changes
        cnt_old = copy.deepcopy(cnt)
        for (v, w) in edges:
            a = cnt[v]
            new_cnt = union(a, cnt_old[w])
            if new_cnt.computeSize() != a.computeSize():
                change = True
            cnt[v] = new_cnt
        harmonic = estimate_centrality(harmonic, cnt, cnt_old, t)
        if not change:
            break
        t += 1
    print("Value comparisons done")
    print("Number of iterations =", t)

    return cnt, harmonic


class FlajoletMartin:
    def __init__(self, bits=19, n_functions=100):
        self.count = 0
        self.bits = bits
        self.modulus = pow(2, self.bits + 1) - 1
        self.max_r = [0 for _ in range(n_functions)]
        self.a = [random.randint(0, self.modulus - 1) for _ in range(n_functions)]
        self.b = [random.randint(0, self.modulus - 1) for _ in range(n_functions)]
        self.alpha_approx = 0.7213 / (1 + 1.079 / pow(2, bits))

    def getEstimatedDistinct(self):
        return [pow(2, max_r) for max_r in self.max_r]

    def hashsing(self, x, idx):
        return (self.a[idx] * int(x) + self.b[idx]) % self.modulus

    def countLeadingZeros(self, x):
        count = 1
        mask = pow(2, self.bits - 1)
        while (x & mask) == 0 and count < self.bits:
            mask = mask >> 1
            count += 1

        return count

    def addElem(self, data):
        for idx in range(len(self.a)):
            count = self.countLeadingZeros(self.hashsing(data, idx))
            if count > self.max_r[idx]:
                self.max_r[idx] = count

    def computeSize(self):
        z = pow(sum([pow(2, -max_r) for max_r in self.max_r]), -1)

        return self.alpha_approx * pow(self.bits, 2) * z


def generate_graph(path="data/test.txt"):
    return nx.read_weighted_edgelist(path, comments="#", create_using=DiGraph)


def plot_graph(graph):
    nx.draw_networkx(graph)
    plt.show()


def main():
    print("Genrating graph...")
    graph = generate_graph()
    n = len(graph.nodes)
    bits = int(math.log(n, 2)) + 1
    print("Graph generated")
    print("Reverting graph...")
    counter, harmonic = hyperball(graph.reverse(), bits=bits)
    real_harmonic = computeHarmonic(graph)
    print(sum([cnt.computeSize() for cnt in counter.values()]) / len(counter))
    print('RMSE is {:.6f}'.format(rmse(real_harmonic, harmonic)))


if __name__ == "__main__":
    main()
