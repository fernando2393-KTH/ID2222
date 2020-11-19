import multiprocessing as mp
import itertools
from time import time

PATH = "results"
PARALLEL = True
CONFIDENCE = 0.6


def association_rules(frequent_items, original_frequent_items):
    items = frequent_items.keys()
    associations = []
    combinations = []
    for item in items:
        combinations.append(list(itertools.combinations(item, len(item) - 1)))

    for idx, item in enumerate(items):
        for pair in combinations[idx]:
            associated2 = set(item) - set(pair)
            num = len(set.intersection(*[set(original_frequent_items[i]) for i in item]))
            denom = len(set.intersection(*[set(original_frequent_items[i]) for i in pair]))
            confidence = num / denom
            if confidence >= CONFIDENCE:
                associations.append(str(pair)+"--->"+str(associated2)+" = "+str(confidence))

    with open(PATH+"/associations.txt", 'w') as f:
        for i in associations:
            f.write(i+"\n")


def check_candidate(original_frequent_items, candidate, min_support):
    intersection = set.intersection(*[set(original_frequent_items[i]) for i in candidate])
    if len(intersection) >= min_support:
        return {candidate: intersection}

    return {}


def pruning(frequent_items, combinatory_factor):
    final_candidates = []
    singletons = set()
    for elem in frequent_items.keys():
        singletons.update(elem)
    candidates = list(itertools.combinations(singletons, combinatory_factor))
    frequent_keys_set = [set(f) for f in frequent_items.keys()]
    for candidate in candidates:
        flag = True
        subset = list(itertools.combinations(candidate, combinatory_factor - 1))
        subset = [set(s) for s in subset]
        for elem in subset:
            if elem not in frequent_keys_set:
                flag = False
                break
        if flag:
            final_candidates.append(candidate)

    return final_candidates


def candidate_k_pairs(frequent_items, original_frequent_items, combinatory_factor, min_support):
    if combinatory_factor == 2:
        candidates = list(itertools.combinations(frequent_items.keys(), combinatory_factor))
    else:
        candidates = pruning(frequent_items, combinatory_factor)
    if not PARALLEL:
        result = []
        for candidate in candidates:
            result.append(check_candidate(original_frequent_items, candidate, min_support))
    else:
        pool = mp.Pool(mp.cpu_count())
        # Parallelize the search of valid candidates
        result = pool.starmap(check_candidate, [(original_frequent_items, c, min_support) for c in candidates])
        pool.close()

    return {k: v for d in result for k, v in d.items()}  # Merge result dictionaries


def main():
    start = time()
    k_tuple = 3  # Maximum size of association
    # Read transactions and baskets' items
    transactions_dict = {}
    n_transactions = 0
    with open("data/T10I4D100K.dat", 'r') as file:
        for i, line in enumerate(file):
            n_transactions += 1
            transaction = [int(num) for num in line.strip().split(' ')]
            for item in transaction:
                if item in transactions_dict.keys():
                    transactions_dict[item].append(i)
                else:
                    transactions_dict[item] = [i]

    # Define variables
    support_threshold = 0.01  # 1% of frequency of the singleton in the total set
    min_support = n_transactions * support_threshold  # Get the minimum amount of basket needed as threshold
    original_frequent_items = dict(filter(lambda elem: len(elem[1]) >= min_support, transactions_dict.items()))
    frequent_items = original_frequent_items
    print("Threshold", min_support)
    print("Amount of frequent items", len(frequent_items))

    # Analyze up to tuples of k elements
    for k in range(2, k_tuple + 1):
        frequent_items = candidate_k_pairs(frequent_items, original_frequent_items, k, min_support)
    association_rules(frequent_items, original_frequent_items)  # Compute association rules for frequent items
    print("Total execution time", time() - start)


if __name__ == "__main__":
    main()
