import numpy as np
import pandas as pd
import sys
import time

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from multiprocessing import Pool

# Constant terms.
STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'label'
DATA_KEY = 'data'
NUM_PROCESSES = 12

def getData():
    print "Getting data matrix..."
    return pd.read_hdf(STORE_NAME, DATA_KEY)

def getLabels():
    print "Getting labels..."
    return pd.read_hdf(STORE_NAME, LABEL_KEY)

def sorted_one_feature_score(data, labels, clf, idx):
    (cases, features) = data.shape
    bucket = features / 10

    cvScores = []
    labels = labels.values

    # Create a multiprocessing pool.
    p = Pool(processes=NUM_PROCESSES)

    for feature in range(idx * bucket, min((idx + 1) * bucket, features)):
        # Select 1 feature at a time.
        oneFeatData = data.iloc[:, feature]

        # cross_val_score only accepts 2D data array.
        thread_data = oneFeatData.values.reshape((cases, 1))
        score = p.apply_async(cv_score_single_feature, args=[labels, clf, thread_data])

        # Need to keep track of index too.
        score_index = (score.get(), feature)
        cvScores.append(score_index)

    # Sort list according to score.
    return sorted(cvScores, key=lambda(score, index):score, reverse=True)

def cv_score_single_feature(labels, classifier, data):
    scaled_data = preprocessing.scale(data)
    scores = cross_val_score(classifier, scaled_data, y=labels, cv=5, scoring='accuracy')
    return np.mean(scores)

def writeShuffledDataToStore(data, labels):
    store = pd.HDFStore(STORE_NAME)
    data.to_hdf(store, 'shuffledData')
    labels.to_hdf(store, 'shuffledLabels')
    store.close()

def one_feature_scoring():

    if len(sys.argv > 1):
        idx = int(sys.argv[1])

    store = pd.HDFStore(STORE_NAME)
    data = getData()
    labels = getLabels()
    store.close()

    print "Shuffling data and labels...\n"
    dataShuffled, labelsShuffled = shuffle(data.T, labels, random_state=0)

    clf = SVC()

    print "Obtaining single feature scores..."
    startTime = time.time()
    oneFeatureCVScores = sorted_one_feature_score(dataShuffled, labelsShuffled, clf, idx)
    endTime = time.time()

    filename = 'one_feature_selection_results_' + str(idx) + ".txt"

    f = open(filename, 'w')
    for i in range(len(oneFeatureCVScores)):
        (score, index) = oneFeatureCVScores[i]
        f.write('Score: %10.8f ; Index: %i' % (score, index))
        f.write("\n")

    f.write('Time taken: %f' % (endTime - startTime))
    f.close()

if __name__ == "__main__":

    if len(sys.argv > 1):
        idx = int(sys.argv[1])

    #idx = 0

    store = pd.HDFStore(STORE_NAME)

    print "Obtaining information from the store..."
    oneFeatureScores = pd.read_hdf(STORE_NAME, 'sorted_one_feature_scores')
    shuffledData = pd.read_hdf(STORE_NAME, 'shuffledData')
    shuffledLabels = pd.read_hdf(STORE_NAME, 'shuffledLabels')
    store.close()

    startTime = time.time()
    (cases, features) = shuffledData.shape
    labels = shuffledLabels.values
    featureIndices = oneFeatureScores.axes[0]
    bestFeature = featureIndices[0]

    bestFeatureValues = shuffledData.iloc[:, bestFeature]
    bestFeatureValues = bestFeatureValues.values.reshape((cases, 1))

    p = Pool(processes=NUM_PROCESSES)
    clf = SVC()
    cvScores = []

    reducedFeatures = features / 2
    bucket = reducedFeatures / 10

    # bucket = 21018, reducedFeatures = 210187
    # maximum idx = 10.
    # In this case, we have [210180, 210187) OR [210180, 210186].
    for i in range(idx * bucket, min((idx + 1) * bucket, reducedFeatures)):

        if idx == 0 and i == 0:
            continue

        secondFeature = shuffledData.iloc[:, featureIndices[i]]
        secondFeature = secondFeature.values.reshape((cases, 1))
        newData = np.column_stack((bestFeatureValues, secondFeature))
        score = p.apply_async(cv_score_single_feature, args=[labels, clf, newData])

        indexScore = (featureIndices[i], score.get())
        cvScores.append(indexScore)

    endTime = time.time()

    filename = 'two_features_selection_results_' + str(idx) + ".txt"

    f = open(filename, 'w')
    for i in range(len(cvScores)):
        (index, score) = cvScores[i]
        f.write('%i ; %f' % (index, score))
        f.write("\n")

    f.write('Time taken: %f' % (endTime - startTime))
    f.close()
