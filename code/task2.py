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

"""
SKETCH
1. Read data
2. Preprocessing Transform each list of instructions into string
	1. token = 1 instruction
	2. Token = 1 mnemonic +suffix
	3. Token = 1 mnemonic
3. Vectorize with TfidfVectorizer or CountVectorizer 
	1. Ngram 1,2
	2. Ngram 1,1
4. Learn with learning algorithm 
	
"""

vectorizers = {
    'tfidf_ngram(1,1)': TfidfVectorizer(ngram_range=(1, 1), tokenizer=tokenize_on_commas),
    'tfidf_ngram(1,2)': TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize_on_commas),
    'count_ngram(1,1)': CountVectorizer(ngram_range=(1, 1), tokenizer=tokenize_on_commas),
    'count_ngram(1,2)': CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize_on_commas),
    'tfidf_ngram(2,3)': TfidfVectorizer(ngram_range=(2, 3), tokenizer=tokenize_on_commas),
    'count_ngram(2,3)': CountVectorizer(ngram_range=(2, 3), tokenizer=tokenize_on_commas),
}

overall_results = {'LSVC': [] ,'MNB': [], 'PT': []}

classifiers = {
    'MNB' : [MultinomialNB(), {'fit_prior': (True, False)}],
    #'LR' : [LogisticRegression(), {'max_iter' : (100,500)}],
    'LSVC': [LinearSVC(), {'max_iter': (100, 200, 400)}],
    # 'DT' : [DecisionTreeClassifier(), {'max_depth':(10, 15)}],
    'PT' : [Perceptron(), {'max_iter': (100, 200)}]
}

# read data
data = pd.read_json('/Users/alessiodevoto/Desktop/Università/machine_learning/homework1/train_dataset.jsonl', lines=True)

# preprocessing data: uncomment option to be used
#data['instructions'] = data['instructions'].apply(','.join)
# apply_short_mnemonics(data)
# apply_mnemonics(data)
apply_extremes(data)

# split into train and test set
x_all_to_vectorize = data['instructions']
y_all = data['compiler']


for vectorizer in vectorizers:
    # vectorize
    x_all = vectorizers[vectorizer].fit_transform(x_all_to_vectorize)

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=15)

    for classifier in classifiers:
        # edit string for saving file
        file_name = vectorizer + '_' + classifier + '.html'
        param_name =str(list(classifiers[classifier][1].keys())[0])

        print('###########################################################################################')
        print('testing with:' + str(classifiers[classifier]))
        print(classifiers[classifier][0])
        print(classifiers[classifier][1])

        grid = GridSearchCV(
            classifiers[classifier][0],
            classifiers[classifier][1],
            refit=True,
            n_jobs=-1,
            scoring='accuracy',
            cv=8,
        )

        test = grid.fit(x_train, y_train)

        print("BEST EST ->")
        print(test.best_estimator_)
        print("<- BEST EST ")

        print("BEST PARAMS ->")
        print(test.best_params_)
        print("<- BEST PARAMS")

        output = pd.DataFrame(
            test.cv_results_,
            index=test.cv_results_['params'],
            columns=['mean_fit_time',
                     'mean_score_time',
                     'param_' + param_name,
                     'mean_test_score',
                     'rank_test_score'])

        display(output)
        with open(file_name, 'w') as results:
            output.to_html(results)
            print("BEST PARAMETERS:\n", file=results)
            print(test.best_params_, file=results)

        x_axis = np.array(test.cv_results_['param_'+param_name].data, dtype=float).tolist()
        y_axis = np.array(test.cv_results_['mean_test_score'].data, dtype=float).tolist()
        for x, y in zip(x_axis, y_axis):
            overall_results[classifier].append([vectorizer, x, y])

comparative_plot(overall_results, classifiers)

