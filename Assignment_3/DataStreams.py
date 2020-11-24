import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing
from networkx import DiGraph
import copy
import math
from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
from time import time


def union(cnt_1, cnt_2):
    new_cnt_1 = copy.deepcopy(cnt_1)
    for i in range(len(cnt_1.max_r)):
        new_cnt_1.max_r[i] = max(cnt_1.max_r[i], cnt_2.max_r[i])  # Compute the maximum of the 2 counters

    return new_cnt_1


def compute_diff(cnt, cnt_old, harmonic, radius, x):
    diff = cnt.computeSize() - cnt_old.computeSize()  # Compute size difference
    harmonic += diff / radius  # Update harmonic for element x

    return {x: harmonic}


def estimate_centrality(harmonic, cnt, cnt_old, radius):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())  # Parallelize centrality computation
    result = pool.starmap(compute_diff, [(cnt[x], cnt_old[x], harmonic[x], radius, x) for x in cnt.keys()])
    pool.close()
    harmonic = {k: v for d in result for k, v in d.items()}

    return harmonic


def computeHarmonic(graph):  # Brute-force harmonic computation
    harmonic = {node: 0 for node in graph.nodes}
    short = {node: 0 for node in graph.nodes}
    for x in tqdm(nx.nodes(graph)):
        for y in nx.nodes(graph):
            if x != y:
                try:
                    shortest_path = nx.shortest_path(graph, y, x)
                    shortest_path = len(shortest_path) - 1
                    short[x] = max(short[x], shortest_path)
                    harmonic[x] += 1 / shortest_path
                except:
                    pass

    return harmonic, short  # Return harmonic and longest shortest path


def rmse(dict1, dict2):
    error = 0
    for k, v in dict1.items():
        error += pow(v - dict2[k], 2)

    return pow((error / len(dict1)), 0.5)


def hyperball(graph, bits, precision):
    print("Graph reverted")
    sizes = []
    edges = [(v, w) for (v, w) in tqdm(graph.edges)]
    cnt = {}
    harmonic = {}
    for node in tqdm(graph.nodes):  # Initialization of counters: 1 per node
        cnt[node] = HyperLogLog(bits=bits, precision=precision)
        cnt[node].addElem(node)
        harmonic[node] = 0
    radius = 1  # Threshold radius
    print("Starting value comparisons...")
    while True:  # While value changes
        change = False  # False --> no value changes
        cnt_old = copy.deepcopy(cnt)
        for (v, w) in edges:
            a = cnt[v]
            new_cnt = union(a, cnt_old[w])
            sizes.append(new_cnt.computeSize())
            if new_cnt.computeSize() != a.computeSize():
                change = True
            cnt[v] = new_cnt
        harmonic = estimate_centrality(harmonic, cnt, cnt_old, radius)
        print("Radius", radius)
        if not change:
            break
        radius += 1
    print("Value comparisons done")

    return cnt, harmonic, radius, sizes


class HyperLogLog:
    def __init__(self, bits, precision):
        self.bits = bits
        self.p = pow(2, self.bits)
        self.precision = precision
        self.modulus = pow(2, self.precision) - 1
        self.max_r = [0 for _ in range(self.p)]

        # Get alpha
        def integral(u):
            return pow(math.log2((2 + u) / (1 + u)), self.p)

        self.alpha_approx = pow(self.p * quad(integral, 0, np.inf)[0], -1)

    @staticmethod
    def hashsing(x):
        return hash(str(x)) & 0xFFFFFFFF  # 32-bit hash of the string value of node ID

    def countLeadingZeros(self, x):
        rho = self.precision - self.bits - x.bit_length() + 1  # Count all positions until 1st one

        if rho <= 0:
            raise ValueError("Overflow")

        return rho

    def rightmost_t_bits(self, number):
        mask_left = pow(2, self.precision - self.bits) - 1
        mask_right = pow(2, self.bits) - 1
        left_num = number >> self.bits & mask_left  # Select left part of the hash
        right_num = number & mask_right  # Select right part of the hash

        return left_num, right_num

    def addElem(self, node):
        hashed_node = self.hashsing(node)
        remaining_bits, i = self.rightmost_t_bits(hashed_node)
        rho_plus = self.countLeadingZeros(remaining_bits)
        self.max_r[i] = max(self.max_r[i], rho_plus)

    def computeEstimate(self, E):  # Error estimation correction
        E_star = 0
        if E <= (5 / 2 * self.p):
            V = len(np.where(np.array(self.max_r) == 0)[0])
            if V != 0:
                E_star = self.p * math.log2(self.p / V)
            else:
                E_star = E
        elif E <= (1 / 30 * pow(2, self.precision)):
            E_star = E
        elif E > (1 / 30 * pow(2, self.precision)):
            E_star = -pow(2, self.precision) * math.log2(1 - E / pow(2, self.precision))

        return E_star

    def computeSize(self):
        z = sum([pow(2, -max_r) for max_r in self.max_r])
        E = self.alpha_approx * pow(self.p, 2) * (1 / z)

        return self.computeEstimate(E)


def generate_graph(path="data/email-Eu-core.txt"):
    return nx.read_edgelist(path, comments="#", create_using=DiGraph)


def plot_graph(graph):
    nx.draw_networkx(graph)
    plt.show()


def main():
    print("Generating graph...")
    graph = generate_graph()
    bits = 5  # based on the paper
    print("N bits: ", bits)
    precision = 32
    print("Graph generated")
    print("Reverting graph...")
    start_approx = time()
    counter, harmonic, radius, sizes = hyperball(graph.reverse(), bits=bits, precision=precision)
    end_approx = time()
    print("Hyperball and HyperlogLog time: ", end_approx - start_approx)
    start_exact = time()
    real_harmonic, short = computeHarmonic(graph)
    end_exact = time()
    print("Exact harmonic time: ", end_exact - start_exact)
    print('RMSE is {:.6f}'.format(rmse(real_harmonic, harmonic)))
    print("Real Longest-Shortest-Path: ", max(short.values()))
    print("Approx. Longest-Shortest-Path: ", radius)
    plt.plot(range(len(sizes)), sizes)
    plt.title("Size approximation per hyperball iteration")
    plt.show()


if __name__ == "__main__":
    main()
