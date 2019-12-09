# usual libraries
import matplotlib
import numpy as np
import pandas as pd
import sklearn as sk
import codecs
import scipy

import seaborn as sns
import matplotlib.pyplot as plt


from aux_functions import *

#read data
data = pd.read_json('/Users/alessiodevoto/Desktop/Università/machine_learning/homework1/train_dataset.jsonl', lines = True)

print("Computing length of each list of instructions\n")
data['len'] = data['instructions'].apply(len)
#data['length'] = data['instructions'].map(lambda text: len(text))

print("Transorming list of instructions into string of instructions.\n")
data['instructions'] = data['instructions'].apply(','.join)

print(data.head())

print("\n\ndata.describe:\n", data.describe())
print("\n\ndata.info:\n", data.info())
print("\n\ndata group by opt: \n", data.groupby('opt').describe())
print("\n\ndata group by opt: \n", data.groupby('compiler').describe())






