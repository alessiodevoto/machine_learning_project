import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



"""AUX FUNCTIONS"""

# reads the first mnemonic_len letter of each instructions and stores them in a string
# this function transforms each list of instructions in a list of the first mnemonic_len letters of each instruction
def get_short_mnemonics(l, mnemonic_len):
   res = ""
   for elem in l:
       res += (elem[:mnemonic_len]) + ","
   return res

def apply_short_mnemonics(dataframe, mnemonic_len = 3):
    for index,row in dataframe.iterrows():
        dataframe['instructions'][index] = get_short_mnemonics(row.instructions, mnemonic_len)


# reads the mnemonic of each instruction and build a string of mnemonics for each instruction
# here we are considering suffixes along with mnemonic
def get_mnemonics(l):
   str1 = ""
   for instruction in l:
       splitted = instruction.split()
       str1 += splitted[0]+ ","
   return str1

def apply_mnemonics(dataframe):
    for index,row in dataframe.iterrows():
        dataframe['instructions'][index] = get_mnemonics(row.instructions)

# transforms each list into a string that contains the first and last 10 instructions, comma separated
def get_extremes(l):
    if(len(l) < 20 ):
        return ','.join(l)
    else:
        return ','.join(l[:9]) +','+ ','.join(l[9:])

def apply_extremes(dataframe):
    for index,row in dataframe.iterrows():
        dataframe['instructions'][index] = get_extremes(row.instructions)

# tokenizer for vectorizers
def tokenize_on_commas(string):
    return string.split(",")

# plot comparative charts
def comparative_plot(results, classifiers):
    for classifier in results:
        # prepare names
        attribute_name = str(list(classifiers[classifier][1].keys())[0])
        file_name = classifier + ".pkl"
        title = classifier

        # build dataframe
        df = pd.DataFrame(results[classifier], columns=['vectorizer', attribute_name , 'score'])
        out_plot = sns.catplot(x=attribute_name, y='score' , hue="vectorizer", kind="bar", legend=True, legend_out=True, data=df);

        plt.subplots_adjust(top=0.9)
        out_plot.fig.suptitle(title)
        plt.show()
        out_plot.savefig(classifier + ".png")
        df.to_pickle(file_name)

# plot confusion matrix (from sklearn official website)
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax




