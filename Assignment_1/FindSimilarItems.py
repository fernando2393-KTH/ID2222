import pandas as pd
import numpy as np
import json
import glob
from CompareSets import CompareSets
import re
import time


def sort_human(file):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    file.sort(key=alphanum)

    return file


def hashing(value):
    return hash(value) & 0xffffffff


def minhashfunction(x, c):
    a = np.random.randint(1, c)
    b = np.random.randint(1, c)
    return (a * x + b) % c  # h(x) = (ax + b) % c


def main():
    # ----- DEFINE VARIABLES ----- #
    json_dict = {}
    k = 5  # Shingle size
    files = sort_human(glob.glob('data/' + '*.json'))
    vocabulary = set()
    permutations = 100
    prime_module = 22751
    np.random.seed(42)

    # ----- IMPORT FILES ----- #
    for idx, file in enumerate(files):
        with open(file, 'r', encoding='utf8') as f:
            # Read text
            json_file = json.load(f)
            json_text = json_file['text']
            # Create and hash shingles (comprehesion)
            shingles = [hashing(json_text[i:i + k]) for i in range(len(json_text) - k + 1)]
            json_dict[idx] = sorted(shingles)
            # Store set of hashed shingles
            vocabulary.update(shingles)

    # Sort vocabulary
    vocabulary = sorted(vocabulary)

    # ----- CREATE BOOLEAN MATRIX ----- #
    boolean_matrix = pd.DataFrame(0, columns=files, index=vocabulary)

    # One-hot enconde matrix based on each document
    for idx, shingles in json_dict.items():
        boolean_matrix.loc[shingles, files[idx]] = 1
    boolean_matrix = boolean_matrix.reset_index(drop=True)  # Resetting hashings to ease operations

    
    # ----- CLASS CompareSets that computes the Jaccard similarity ----- #
    set_comparator = CompareSets()
    set_comparator.set_boolean_matrix(boolean_matrix)
    _ = set_comparator.jaccardSimilarity(files=files, heatmap=True)


    # ----- MIN-HASHING ----- #
    # Perform Min-Hashing and define the Signature Matrix to reduce the size of the matrix
    signature_matrix = pd.DataFrame(0, columns=files, index=np.arange(permutations))  # Number of permutations x
    # Number of documents
    files_array = np.array(files)

    for permutation in range(permutations):
        # Copying original boolean and shuffling it
        aux_matrix = boolean_matrix.copy(deep=True)
        hashed_indices = minhashfunction(aux_matrix.index, prime_module)
        aux_matrix = aux_matrix.reindex(hashed_indices).reset_index(drop=True)
        # Iterate through the rows of the new shuffled boolean matrix
        for index, row in aux_matrix.iterrows():
            # Check if all of the columns in the Signature matrix have their shingle hash,
            # otherwise check next row of the shuffled boolean matrix
            if np.count_nonzero(signature_matrix.loc[permutation].values == 0) == 0:
                break
            else:
                # Check the documents that have the current shingle
                match = np.where((row == 1) & (signature_matrix.loc[permutation].values == 0))
                if match[0] is not None:
                    # Assign the shingle to the positions that haven't been assigned to any shingle before
                    signature_matrix.loc[permutation, files_array[match[0]]] = index

    signature_matrix = signature_matrix.iloc[::-1].reset_index(drop=True)  # Flip matrix in order
    # to match the theory requirements
    set_comparator.set_signature_matrix(signature_matrix)
    _ = set_comparator.signatureSimilarity(permutations, files=files, heatmap=True)


    # ----- CLASS CompareSignatures that estimates similarity of two integer vectors – minhash signatures ----- #
    _ = set_comparator.lshashing(permutations, files, bands=20)


if __name__ == "__main__":
    main()
