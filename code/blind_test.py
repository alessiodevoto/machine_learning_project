# usual libraries
import numpy as np
import pandas as pd
import sklearn as sk
import codecs
import scipy

# libraries for charts
import seaborn as sns
import matplotlib.pyplot as plt

# feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# learning algorithms
from sklearn.naive_bayes import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier

# functions to work with pandas dataframe
from aux_functions import *
from sklearn.model_selection import train_test_split

# evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

from IPython.display import display

''' BLIND DATASET'''
# read blind dataset
data_blind = pd.read_json('/Users/alessiodevoto/Desktop/Università/machine_learning/homework1/test_dataset_blind.jsonl', lines=True)

# preprocess lines
data_blind['instructions'] = data_blind['instructions'].apply(','.join)

# vectorize lines
vect1 = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize_on_commas)
x_blind = vect1.fit_transform(data_blind['instructions'])

# now we have to train a classifier based on the train set

''' GROUND TRUTH '''
# read train dataset
data = pd.read_json('/Users/alessiodevoto/Desktop/Università/machine_learning/homework1/train_dataset.jsonl', lines=True)

# preprocess lines
data['instructions'] = data['instructions'].apply(','.join)

# vectorize lines
x = vect1.transform(data['instructions'])

# prepare target for task 1 and 2
y1 = data['opt']
y2 = data['compiler']

# instantiate classifier for task1 and fit model
clf1 = LinearSVC(max_iter = 100)
clf1.fit(x, y1)

# instantiate classifier for task2 and fit model
clf2 = LinearSVC(max_iter = 100)
clf2.fit(x, y2)

# create new empty dataset
data_blind2 = pd.DataFrame(columns = ['opt', 'compiler'])

# now we have a classifier ready to predict on the blind test set
# we predict y1pred and add it to blind data
y1_blind = clf1.predict(x_blind)
print(y1_blind)
data_blind2['opt'] = y1_blind

# we predict y1pred and add it to blind data
y2_blind = clf2.predict(x_blind)
print(y2_blind)
data_blind2['compiler'] = y2_blind


print(data_blind2.head())
data_blind2.to_csv('test0.csv', index = False)

