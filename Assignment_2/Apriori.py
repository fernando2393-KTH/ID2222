import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import itertools
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os
from os import path
from time import time

PATH = "results"
CONFIDENCE = 0.6


def save_items(items, dict_name):
    if path.exists(PATH):
        pickle.dump(items, open(PATH + "/" + dict_name + ".p", "wb"))
    else:
        os.mkdir(PATH)
        pickle.dump(items, open(PATH + "/" + dict_name + ".p", "wb"))


def association_rules(items, boolean_matrix):
    associations = []
    combinations = []
    for item in items:
        combinations.append(list(itertools.combinations(item, 2)))

    for i in range(len(items)):
        for pair in combinations[i]:
            associated2 = set(items[i]) - set(pair)
            num = len(np.where(boolean_matrix[list(items[i])].sum(axis=1) == len(items[i]))[0])
            denom = len(np.where(boolean_matrix[list(pair)].sum(axis=1) == len(pair))[0])
            confidence = num / denom
            if confidence >= CONFIDENCE:
                associations.append(str(pair)+"--->"+str(associated2)+" = "+str(confidence))

    with open(PATH+"/associations.txt", 'w') as f:
        for i in associations:
            f.write(i+"\n")


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
    pool = mp.Pool(mp.cpu_count())
    start = time()
    result = pool.starmap(check_candidate, [(c, min_support, boolean_matrix) for c in candidates])
    end = time()
    print("Time required for parallelization ", end - start)
    pool.close()

    return {k: v for d in result for k, v in d.items()}


def main():
    k_tuple = 3
    frequent_items = None
    boolean_matrix = None
    try:
        dict_list = []
        for i in range(k_tuple):
            dict_list.append(pickle.load(open(PATH+"/"+str(i + 1)+".p", "rb")))
        frequent_items = dict_list[-1]
        boolean_matrix = pickle.load(open(PATH+"/boolean_matrix.p", "rb"))

    except:
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
        save_items(boolean_matrix, "boolean_matrix")
        print("Amount of frequent items", len(frequent_items))
        save_items(frequent_items, "1")

        # Analyze up to k-tuples
        for k in range(2, k_tuple + 1):
            frequent_items = candidate_k_pairs(frequent_items, k, min_support, boolean_matrix)
            save_items(frequent_items, str(k))
            print(frequent_items)

    finally:
        association_rules(list(frequent_items.keys()), boolean_matrix)


if __name__ == "__main__":
    main()
