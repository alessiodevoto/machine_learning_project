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

# functions to work with pandas dataframe
from aux_functions import *
from sklearn.model_selection import train_test_split

# evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

from IPython.display import display

""" TEST FOR BINARY CLASSIFICATION """

data1 = pd.read_json('/Users/alessiodevoto/Desktop/Università/machine_learning/homework1/train_dataset.jsonl', lines=True)
data2 = data1.copy()

# best conversion method
data1['instructions'] = data1['instructions'].apply(','.join)

# best classifier
clf1 = LinearSVC(max_iter=200)

# best vectorizer
vect1 = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize_on_commas)
x_all1 = vect1.fit_transform(data1['instructions'])
y_all1 = data1['opt']

# train test split
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_all1, y_all1, test_size=0.2, random_state=15)

# fit model
clf1.fit(x_train1, y_train1)
y_pred1 = clf1.predict(x_test1)

# print confusion matrix
#names =np.asarray([0, 1])
names = ['H', 'L']
plot_confusion_matrix(y_test1, y_pred1, classes = names, normalize = True )
np.set_printoptions(precision=2)
plt.show()
plt.savefig("confusion1.png")

# print classification report
clf_report = classification_report(y_test1, y_pred1)
print(clf_report)

""" TEST FOR MULTICLASS CLASSIFICATION """

#best conversion method
data2['instructions'] = data2['instructions'].apply(','.join)

#best classifier
clf2 = LinearSVC(max_iter = 200)

# best vectorizer
vect2 = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize_on_commas)
x_all2 = vect1.fit_transform(data2['instructions'])

# same as before
y_all2 = data2['compiler']

# train test split
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_all2, y_all2, test_size=0.2, random_state=15)

# fit model
clf2.fit(x_train2, y_train2)
y_pred2 = clf2.predict(x_test2)

# print confusion matrix
#names =np.asarray([0, 1])
names = ['gcc', 'icc', 'Clang']
plot_confusion_matrix(y_test2, y_pred2, classes = names, normalize = True )
plt.show()

# print classification report
clf_report = classification_report(y_test2, y_pred2)
print(clf_report)
