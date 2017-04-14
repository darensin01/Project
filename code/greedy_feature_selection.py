import numpy as np
import pandas as pd
import sys
import time

from ast import literal_eval as make_tuple
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from multiprocessing import Pool

# Constant terms.
STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'label'
DATA_KEY = 'data'
NUM_PROCESSES = 1
ITERATION = 0

def getData():
    print "Getting data matrix..."
    return pd.read_hdf(STORE_NAME, DATA_KEY)

def getLabels():
    print "Getting labels..."
    return pd.read_hdf(STORE_NAME, LABEL_KEY)


def cv_score_single_feature(labels, classifier, data):
    scaled_data = preprocessing.scale(data)
    scores = cross_val_score(classifier, scaled_data, y=labels, cv=5, scoring='accuracy')
    return np.mean(scores)

def writeShuffledDataToStore(data, labels):
    store = pd.HDFStore(STORE_NAME)
    data.to_hdf(store, 'shuffledData')
    labels.to_hdf(store, 'shuffledLabels')
    store.close()

def getBestIndicesFromFiles():
    allPreviousBestFeatures = []

    # Iterates from [2, ITERATION - 1].
    # Get best paired feature from previous results file.
    # Reads just the first line of the file and appends it to the list.
    for i in range(2, ITERATION):
        print i
        filename = "all_" + str(i) + "_feature_results.txt"
        f = open(filename, 'r')
        firstLine = f.readline()
        allPreviousBestFeatures.append(make_tuple(firstLine)[0])
        f.close()

    return allPreviousBestFeatures

if __name__ == "__main__":

    store = pd.HDFStore(STORE_NAME)

    print "Obtaining information from the store..."
    oneFeatureScores = pd.read_hdf(STORE_NAME, 'sorted_one_feature_scores')
    shuffledData = pd.read_hdf(STORE_NAME, 'shuffledData')
    shuffledLabels = pd.read_hdf(STORE_NAME, 'shuffledLabels')
    store.close()

    startTime = time.time()
    (cases, features) = shuffledData.shape
    labels = shuffledLabels.values

    # Get all the ranked feature scores.
    featureIndices = oneFeatureScores.axes[0]

    # Find best feature from one_feature_scores.
    # Find previous best scores from files, and append best feature to front of
    # the list.
    bestFeature = featureIndices[0]
    otherFeatures = getBestIndicesFromFiles()
    otherFeatures.insert(0, bestFeature)

    bestFeatureValues = shuffledData.iloc[:, otherFeatures]
    bestFeatureValues = bestFeatureValues.values.reshape((cases, ITERATION - 1))

    p = Pool(processes=NUM_PROCESSES)
    clf = SVC()
    cvScores = []

    reducedFeatures = features / ITERATION
    bucket = reducedFeatures / 3

    for i in range(idx * bucket, min((idx + 1) * bucket, reducedFeatures)):
        # We check if the ith feature is already part of the list.
        if featureIndices[i] in otherFeatures:
            continue

        # Concatenate the feature to be tested, with the rest of the previous
        # indices, by using column_stack.
        featureToBeTested = shuffledData.iloc[:, featureIndices[i]]
        featureToBeTested = featureToBeTested.values.reshape((cases, 1))
        newData = np.column_stack((bestFeatureValues, featureToBeTested))

        # Find the cross validation score.
        score = p.apply_async(cv_score_single_feature, args=[labels, clf, newData])

        indexScore = (featureIndices[i], score.get())
        print indexScore
        cvScores.append(indexScore)

    endTime = time.time()

    print "Writing to file..."
    name = str(ITERATION) + '_feature_selection_results_' + str(idx) + ".txt"

    f = open(name, 'w')
    for i in range(len(cvScores)):
        (index, score) = cvScores[i]
        f.write('%i ; %10.8f' % (index, score))
        f.write("\n")

    f.write('Time taken: %f' % (endTime - startTime))
    f.close()
