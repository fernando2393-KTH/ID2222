import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

    def lshashing(self, files, bands=10):
        heatmap_matrix = np.zeros((self.signature_matrix.shape[1], self.signature_matrix.shape[1]))
        buckets = dict()
        rows = int(self.signature_matrix.shape[0] / bands)
        threshold = pow((1 / bands), (1 / rows))

        for b in range(bands):
            band = self.signature_matrix[rows * b:rows * (b + 1)]
            aux_bucket = dict()
            for i in range(band.shape[1]):
                lst = [files[i]]
                for j in range(i + 1, band.shape[1]):
                    val = compute_signature_similarity(band, files[i], files[j], rows)
                    if val >= threshold:
                        heatmap_matrix[i, j] += 1
                        heatmap_matrix[j, i] += 1
                        lst.append(files[j])
                aux_bucket[i] = lst
            buckets[b] = aux_bucket

        sns.heatmap(heatmap_matrix, annot=True, cmap="YlGnBu")
        plt.show()

        return buckets
