import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import itertools
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer


def check_candidate(candidate, min_support, boolean_matrix):
    candidate_items = {}
    compare_columns = len(np.where(boolean_matrix[list(candidate)].sum(axis=1) == len(candidate))[0])
    if compare_columns >= min_support:
        candidate_items[candidate] = compare_columns

    return candidate_items


def item_counts(data, min_support):
    transaction_set = []
    # Flatten items sets
    for row, transaction in tqdm(data.iterrows()):
        transaction_set.extend(transaction[0])
    mlb = MultiLabelBinarizer()
    boolean_matrix = pd.DataFrame(mlb.fit_transform(data["basket"]),
                                  columns=mlb.classes_,
                                  index=data.index)
    # Count the items
    item_cnt = Counter(transaction_set)
    # Filter items that are not frequent
    frequent_items = dict(filter(lambda elem: elem[1] >= min_support, item_cnt.items()))

    return frequent_items, boolean_matrix


def candidate_k_pairs(frequent_items, combinatory_factor, min_support, boolean_matrix):
    if combinatory_factor == 2:
        keys = list(frequent_items.keys())
    else:
        keys = Counter(sum(list(k for k in frequent_items.keys()), ()))
        keys = list(dict(filter(lambda elem: elem[1] >= combinatory_factor - 1, keys.items())).keys())

    candidates = list(itertools.combinations(keys, combinatory_factor))
    num_cores = multiprocessing.cpu_count()
    result = Parallel(n_jobs=num_cores)(delayed(check_candidate)(c, min_support, boolean_matrix)
                                        for c in tqdm(candidates))

    return {k: v for d in result for k, v in d.items()}


def main():
    # Read transactions and baskets' items
    transactions = pd.read_csv("data/T10I4D100K.dat", header=None, names=["basket"])
    transactions.index.names = ['transaction']
    transactions["basket"] = transactions["basket"].str.split()

    # Define variables
    n_transactions = transactions.shape[0]
    support_threshold = 0.01  # 1% of frequency of the singleton in the total set
    min_support = n_transactions * support_threshold

    print("Threshold", min_support)
    frequent_items, boolean_matrix = item_counts(transactions, min_support)
    print("Amount of frequent items", len(frequent_items))

    # Analyze up to 3-tuples
    k_tuple = 3
    for k in range(2, k_tuple + 1):
        frequent_items = candidate_k_pairs(frequent_items, k, min_support, boolean_matrix)
        print(frequent_items)


if __name__ == "__main__":
    main()
