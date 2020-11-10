import numpy as np
import pandas as pd
import itertools
from collections import Counter
import time
import multiprocessing
from joblib import Parallel, delayed

# Read transactions and baskets' items
transactions = pd.read_csv("data/T10I4D100K.dat", header=None, names=["basket"])
transactions.index.names = ['transaction']

# Define variables
n_transactions = transactions.shape[0]
support_threshold = 0.01 # 1% of frequency of the singleton in the total set
min_support = n_transactions * support_threshold


def item_counts(data, min_support):
    item_counts = {}
    transaction_set = []

    # Flatten items sets
    for row, transaction in data.iterrows():
        transaction_set.extend(transaction[0].split())
    
    # Count the items
    item_counts = Counter(transaction_set)

    # Filter items that are not frequent
    frequent_items = dict(filter(lambda elem: elem[1] >= min_support, item_counts.items()))

    return frequent_items

print("Threshold", min_support)
frequent_items = item_counts(transactions, min_support)
print("Amount of frequent items", len(frequent_items))

def candidate_k_pairs(data, frequent_items, combinatory_factor, min_support):
   elem = set(list(sum(frequent_items.keys(), ())))


# Analyze up to 3-tuples
k_tuple = 3
for k in range(1, k_tuple):
    frequent_items = candidate_k_pairs(transactions, frequent_items, k, min_support)
    print(frequent_items)
    break