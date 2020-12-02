import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
import scipy.linalg as la
from sklearn.cluster import KMeans


def load_graph(path):
    if path == "data/example1.dat":
        G = nx.read_edgelist(path, delimiter=",", create_using=DiGraph)
    else:
        G = nx.read_weighted_edgelist(path, delimiter=",", create_using=DiGraph)

    return G


def plot_graph(graph, labels, pos):
    nx.draw_networkx(graph, pos=pos, node_size=6, node_color=labels,
                           cmap=plt.cm.Set1, with_labels=False)
    plt.show()


def plot_eigen_values_diff(eig_values):
    plt.scatter(range(len(eig_values)), eig_values)
    plt.title("Biggest eigen values difference to find K-clusters")
    plt.show()


def plot_fiedler(f):
    plt.plot(range(len(f)), f)
    plt.title("Sorted Fiedler Vector")
    plt.show()


def plot_sparse(A):
    plt.imshow(A, cmap='Blues', interpolation='nearest')
    plt.show()


def affinity_matrix(graph):
    """
    A (n x n)
    A_ii = 0
    A_ij = exp(-|s_i - s_j|^2 / (2 * sigma^2))
    """

    A = np.asarray(nx.adjacency_matrix(graph).todense())

    return A


def L_matrix(A):
    """
    L and D (n x n)
    D = diagonal matrix with D_ii = sum of affinity matrix i_th row
    L = D ^ (-1/2) * A * D ^ (-1/2)
    """
    D = np.diag(np.sum(A, axis=1))
    D_inv = np.linalg.inv(np.sqrt(D))
    L = D_inv @ A @ D_inv

    return L


def compute_k(eigen_values):
    eigen_values_diff = np.diff(eigen_values)
    index_largest_diff = np.argmax(eigen_values_diff) + 1
    k = len(eigen_values) - index_largest_diff
    return k


def eigenvector_matrix(L):
    """
    X (n x k)
    k largest eigenvectors of L orthogonal from each other
    """
    X_values, X_vectors = la.eigh(L)
    k = compute_k(X_values)
    for i in range(1, k+2):
        plt.plot(sorted(X_vectors[:, -i]))
    plt.show()

    fiedler_vec = sorted(X_vectors[:, -2])
    X = X_vectors[:, -k:]

    return X, k, X_values, fiedler_vec


def normalized_X(X):
    """
    Y (n x k)
    Y = X_ij / (Sum_j (X_ij ^ 2)) ^(1/2)
    """
    Y = X / np.sqrt(np.sum(pow(X, 2), axis=1)).reshape((-1, 1))

    return Y


def main():
    path = "data/example1.dat"
    graph = load_graph(path)
    print("Number of nodes: ", len(graph.nodes))
    pos = nx.spring_layout(graph)
    A = affinity_matrix(graph)
    L = L_matrix(A)
    X, k, eigen_values, fiedler_vec = eigenvector_matrix(L)
    Y = normalized_X(X)
    clustering = KMeans(n_clusters=k).fit(Y)
    labels = clustering.labels_
    plot_graph(graph, labels, pos)
    plot_eigen_values_diff(eigen_values)
    plot_fiedler(fiedler_vec)
    plot_sparse(A)


if __name__ == "__main__":
    main()
