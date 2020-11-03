import pandas as pd
import numpy as np
import json
import glob
import time

def hashing(value):
    return hash(value) & 0xffffffff

# ----- DEFINE VARIABLES ----- #
json_dict = {}
json_file = None
k = 10 # shingle size
files = glob.glob('data/' + '*.json')
vocabulary = set()
permutations = 100
np.random.seed(42)

# ----- IMPORT FILES ----- #
for idx, file in enumerate(files):
    with open(file, 'r', encoding='utf8') as f:
        # Read text
        json_file = json.load(f)
        json_text = json_file['text']

        # Create and hash shingles (compression)
        shingles = [hashing(json_text[i:i+k]) for i in range(len(json_text) - k + 1)]

        json_dict[idx] = shingles

        # Store set of hashed shingles
        vocabulary.update(shingles)

# Sort vocabulary
vocabulary = sorted(vocabulary)

# ----- CREATE BOOLEAN MATRIX ----- #

boolean_matrix = pd.DataFrame(0, columns=files, index=vocabulary)

# One-hot enconde matrix based on each document
for idx, shingles in json_dict.items():
    mask = sorted(shingles)
    boolean_matrix.loc[mask, files[idx]] = 1

# ----- CLASS CompareSets that computes the Jaccard similarity ----- #

# ----- MIN-HASHING ----- #
# Perform Min-Hashing and define the Signature Matrix to reduce the size of the matrix
signature_matrix = pd.DataFrame(0, columns=files, index=np.arange(permutations)) # Number of permutations x Number of documents
files_array = np.array(files)

for permutation in range(permutations):
    # Copying original boolean and shuffling it 
    aux_matrix = boolean_matrix.copy(deep= True)
    aux_matrix.index = np.random.permutation(aux_matrix.index)

    # Iterate through the rows of the new shuffled boolean matrix
    for index, row in aux_matrix.iterrows():
        # Check if all of the columns in the Signature matrix have their shingle hash, otherwise check next row of the shuffled boolean matrix
        if np.count_nonzero(signature_matrix.loc[permutation].values == 0) == 0:
            break
        else:
            # Check the documents that have the current shingle
            match = np.where((row == 1) & (signature_matrix.loc[permutation].values == 0))
            if match[0] is None:
                pass  
            else:
                # Assign the shingle to the positions that haven't been assigned to any shingle before 
                signature_matrix.loc[permutation, files_array[match[0]]] = index

print(signature_matrix.head())

# ----- CLASS CompareSignatures that estimates similarity of two integer vectors – minhash signatures ----- #
