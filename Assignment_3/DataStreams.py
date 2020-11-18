import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
import random
import copy
from tqdm import tqdm


def union(cnt_1, cnt_2):
    for i in range(len(cnt_1.max_r)):
        cnt_1.max_r[i] = max(cnt_1.max_r[i], cnt_2.max_r[i])

    return cnt_1


def estimate_centrality(harmonic, cnt, cnt_old, t):
    for node in cnt.keys():
        harmonic[node] += (cnt[node].computeSize() - cnt_old[node].computeSize()) / t

    return harmonic


def hyperball(graph):
    print("Graph reverted")
    nodes = [int(node) for node in graph.nodes]
    edges = [(int(v), int(w)) for (v, w) in graph.edges]
    cnt = {node: FlajoletMartin() for node in nodes}  # Initialize a fjm counter for each node
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
            change |= not (new_cnt.computeSize() == a.computeSize())
            cnt[v] = new_cnt
        harmonic = estimate_centrality(harmonic, cnt, cnt_old, t)
        if not change:
            break
        t += 1
    print("Value comparisons done")

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
        return (self.a[idx] * x + self.b[idx]) % self.modulus

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


def generate_graph(path="data/web-Stanford.txt"):
    return nx.read_edgelist(path, comments="#", create_using=DiGraph)


def plot_graph(graph):
    nx.draw_networkx(graph)
    plt.show()


def main():
    print("Genrating graph...")
    graph = generate_graph()
    print("Graph generated")
    print("Reverting graph...")
    counter, harmonic = hyperball(graph.reverse())
    print(sum([cnt.computeSize() for cnt in counter.values()]) / len(counter))


if __name__ == "__main__":
    main()
