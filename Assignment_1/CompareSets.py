import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def hashing(value):
    return hash(value) & 0xffffffff


def compute_jaccard_similarity(boolean_matrix, doc1, doc2):
    col1 = boolean_matrix.loc[:, doc1].to_numpy()
    col2 = boolean_matrix.loc[:, doc2].to_numpy()
    numerator = np.sum(col1 & col2)
    denominator = np.sum(col1 | col2)

    return numerator / denominator


def compute_signature_similarity(signature_matrix, doc1, doc2, permutations):
    col1 = signature_matrix.loc[:, doc1].to_numpy()
    col2 = signature_matrix.loc[:, doc2].to_numpy()
    numerator = np.sum(col1 == col2)
    denominator = permutations

    return float(numerator / denominator)


def signature_hashing(array):
    stringified_val = ",".join([str(c) for c in array])

    return hashing(stringified_val)


class CompareSets:
    def __init__(self):
        self.boolean_matrix = None
        self.signature_matrix = None

    def set_boolean_matrix(self, boolean_matrix):
        self.boolean_matrix = boolean_matrix

    def set_signature_matrix(self, signature_matrix):
        self.signature_matrix = signature_matrix

    def jaccardSimilarity(self, doc1=None, doc2=None, files=None, heatmap=False):
        if heatmap:
            heatmap_matrix = np.ones((self.boolean_matrix.shape[1], self.boolean_matrix.shape[1]))
            for i in range(heatmap_matrix.shape[0]):
                for j in range(i + 1, heatmap_matrix.shape[1]):
                    val = compute_jaccard_similarity(self.boolean_matrix, files[i], files[j])
                    heatmap_matrix[i, j] = val
                    heatmap_matrix[j, i] = val

            sns.heatmap(heatmap_matrix, annot=True, cmap="YlGnBu")
            plt.show()

            return heatmap_matrix

        else:
            return compute_jaccard_similarity(self.boolean_matrix, doc1, doc2)

    def signatureSimilarity(self, permutations, doc1=None, doc2=None, files=None, heatmap=False):
        if heatmap:
            heatmap_matrix = np.ones((self.signature_matrix.shape[1], self.signature_matrix.shape[1]))
            for i in range(heatmap_matrix.shape[0]):
                for j in range(i + 1, heatmap_matrix.shape[1]):
                    val = compute_signature_similarity(self.signature_matrix, files[i], files[j], permutations)
                    heatmap_matrix[i, j] = val
                    heatmap_matrix[j, i] = val

            sns.heatmap(heatmap_matrix, annot=True, cmap="YlGnBu")
            plt.show()

            return heatmap_matrix

        else:
            return compute_signature_similarity(self.signature_matrix, doc1, doc2, permutations)

    def lshashing(self, permutations, files, bands=10, n_buckets=20):
        heatmap_matrix = np.zeros((self.signature_matrix.shape[1], self.signature_matrix.shape[1]))
        candidates = list()
        rows = int(self.signature_matrix.shape[0] / bands)
        # threshold = pow((1 / bands), (1 / rows))
        threshold = 0.2

        for b in range(bands):
            buckets = {k: [] for k in range(n_buckets)}
            band = self.signature_matrix[rows * b:rows * (b + 1)]
            hashes = [signature_hashing(band.loc[:, files[i]]) for i in range(band.shape[1])]
            modules = np.array(hashes) % n_buckets
            for idx, elem in enumerate(modules):
                if files[idx] not in buckets[elem]:
                    buckets[elem].append(files[idx])
            for elem in list(buckets.values()):
                if len(elem) > 1:
                    candidates.append(elem)

        for candidate in candidates:
            combinations = list(itertools.combinations(candidate, 2))
            for combination in combinations:
                i = self.signature_matrix.columns.get_loc(combination[0])
                j = self.signature_matrix.columns.get_loc(combination[1])
                if heatmap_matrix[i, j] == 0:
                    val = compute_signature_similarity(self.signature_matrix, combination[0], combination[1],
                                                       permutations)
                    heatmap_matrix[i, j] = val
                    heatmap_matrix[j, i] = val

        heatmap_matrix = (heatmap_matrix > threshold) * 1

        sns.heatmap(heatmap_matrix, annot=True, cmap="YlGnBu")
        plt.show()

        return heatmap_matrix
