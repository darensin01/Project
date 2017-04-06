import numpy as np
import pandas as pd
import sys
import time

from ast import literal_eval as make_tuple
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from multiprocessing import Pool
from sklearn.metrics import mutual_info_score

# Constant terms.
STORE_NAME = 'DataStore.h5'
SHUFFLED_LABEL_KEY = 'shuffledLabels'
SHUFFLED_DATA_KEY = 'shuffledData'
NUM_PROCESSES = 1

def getData():
    print "Getting data matrix..."
    return pd.read_hdf(STORE_NAME, SHUFFLED_DATA_KEY)

def getLabels():
    print "Getting labels..."
    return pd.read_hdf(STORE_NAME, SHUFFLED_LABEL_KEY)

if __name__ == "__main__":

    store = pd.HDFStore(STORE_NAME)

    print "Obtaining information from the store..."
    shuffledData = pd.read_hdf(STORE_NAME, SHUFFLED_DATA_KEY)
    shuffledLabels = pd.read_hdf(STORE_NAME, SHUFFLED_LABEL_KEY)
    store.close()

    (cases, features) = shuffledData.shape
    labels = shuffledLabels.values

    # Get entropy of target labels.
    count_one = sum(labels)
    count_zero = cases - count_one

    prob_one = count_one / float(cases)
    prob_zero = count_zero / float(cases)
    entropy = -prob_one * np.log2(prob_one) - prob_zero * np.log2(prob_zero)

    print entropy

    startTime = time.time()

    array = [258645, 251478, 369209, 49346, 92298]

    for i in array:
        feature = shuffledData.iloc[:, i]
        scaled_feature = preprocessing.scale(feature)
        mean = np.mean(feature)
        std = np.std(feature)

        below_std = mean - std
        above_std = mean + std

        for j in range(feature.shape[0]):
            if feature[j] >= above_std:
                feature[j] = 1
            elif feature[j] <= below_std:
                feature[j] = -1
            else:
                feature[j] = 0

        score = mutual_info_score(labels, feature)
        #print score

    endTime = time.time()

    print "Writing to file..."
    name = 'mi_scores.txt'

    '''
    f = open(name, 'w')
    for i in range(len(cvScores)):
        (index, score) = cvScores[i]
        f.write('%i ; %10.8f' % (index, score))
        f.write("\n")

    f.write('Time taken: %f' % (endTime - startTime))
    f.close()
    '''