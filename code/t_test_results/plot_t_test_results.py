from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr

import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

NUM_FEATURES = 420374

# Constant terms.
STORE_NAME = '..\DataStore.h5'
LABEL_KEY = 'shuffledLabels'
DATA_KEY = 'shuffledData'

if __name__ == "__main__":

    results = []

    filename = "t_test_results_unequal_var.txt"
    f = open(filename, 'r')

    print "Reading file..."

    for i in xrange(2):
        content = f.readline().strip()
        (index, p_val) = make_tuple(content)
        results.append((index, p_val))

    f.close()

    results = list(reversed(results))

    print "Reading from store..."

    # Read from data store.
    store = pd.HDFStore(STORE_NAME)
    shuffledData = pd.read_hdf(STORE_NAME, DATA_KEY)
    shuffledLabels = pd.read_hdf(STORE_NAME, LABEL_KEY)
    store.close()

    (cases, features) = shuffledData.shape
    labels = shuffledLabels.values

    indices = map(lambda (idx, score):idx, results)

    '''
    MAXIMUM = 7
    OFFSET = 2000

    corr = np.zeros((MAXIMUM, MAXIMUM))
    for i in range(OFFSET, OFFSET + MAXIMUM):
        index_first = indices[i]
        feature_first = preprocessing.scale(shuffledData.iloc[:, index_first])
        for j in range(i, OFFSET + MAXIMUM):
            index_second = indices[j]
            feature_second = preprocessing.scale(shuffledData.iloc[:, index_second])
            corr[i - OFFSET, j - OFFSET] = pearsonr(feature_first, feature_second)[0]

    print corr
    '''

    # Select top i faetures from sorted list to fit into classifier.
    # Then, obtain cross validation scores for 5-fold cross validation.
    RANGE = range(1589, 1590)
    result_scores = []

    for i in RANGE:
        print "Current index: %i" % i
        topResults = [results[j] for j in range(i)]
        topIdx = map(lambda (idx, p_value):idx, topResults)

        clf = SVC(kernel='linear')
        selected_features = shuffledData.iloc[:, topIdx]
        selected_features = selected_features.apply(preprocessing.scale)

        cross_val_scores = cross_val_score(clf, selected_features, y=labels, cv=5, scoring='accuracy')
        result_scores.append(np.mean(cross_val_scores))

    max_idx = result_scores.index(max(result_scores))
    print RANGE[max_idx]

    # Plot graph
    linear_max = 0.929160
    plt.plot(RANGE, result_scores)
    plt.plot(RANGE, [linear_max] * len(RANGE), 'k--', linewidth=3)
    plt.xlabel("Number of features", fontsize=25)
    plt.ylabel("Cross validation (CV) score", fontsize=25)
    plt.title("CV score for features selected by t-test using SVM with linear kernel", fontsize=25)
    plt.show()

