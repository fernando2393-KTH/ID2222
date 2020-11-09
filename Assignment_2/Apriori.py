import numpy as np
import pandas as pd

df = pd.read_csv("data/T10I4D100K.dat", header=None, names=["basket"])
print(df.head())