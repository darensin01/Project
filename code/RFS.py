import numpy as np
import pandas as pd
import time

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from multiprocessing import Pool

# Constant terms.
STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'shuffledLabels'
DATA_KEY = 'shuffledData'
NUM_PROCESSES = 4
K = 100

def getData():
    return pd.read_hdf(STORE_NAME, DATA_KEY)

def getLabels():
    return pd.read_hdf(STORE_NAME, LABEL_KEY)

def getCVScore(labels, classifier, data):
    scaled_data = preprocessing.scale(data)
    scores = cross_val_score(classifier, scaled_data, y=labels, cv=5, scoring='accuracy')
    return np.mean(scores)

if __name__ == "__main__":

    store = pd.HDFStore(STORE_NAME)

    shuffledData = getData()
    shuffledLabels = getLabels()
    store.close()

    (cases, features) = shuffledData.shape
    labels = shuffledLabels.values

    p = Pool(processes = NUM_PROCESSES)
    clf = SVC(kernel='linear')

    singleCVIndexScore = []

    startTime = time.time()

   # Evaluate CV score of each feature singly.
    for i in range(features):
        feature = shuffledData.iloc[:, i].values.reshape((cases, 1))
        cv_score = p.apply_async(getCVScore, args=[labels, clf, feature])
        indexScore = (i, cv_score.get())
        singleCVIndexScore.append(indexScore)

    # Sort the (index, score) list in decreasing order.
    singleCVIndexScore = sorted(singleCVIndexScore, key=lambda (index, score):score, reverse=True)

    # Create a list of the indices of the sorted scores.
    singleCVIndex = [indexScore[0] for indexScore in singleCVIndexScore]

    # Keep track of the index corresponding to the best CV score.
    selectedFeatureIndex = [singleCVIndex[0]]

    # Also, keep track of the best CV score.
    selectedFeatureScores = [singleCVIndexScore[0][1]]

    # Get K features.
    for i in range(2, K+1):

        # Keep track of the CV scores for this iteration.
        indexScores = []

        # Construct a list of feature values corresponding to the best index.
        selectedFeatures = shuffledData.iloc[:, selectedFeatureIndex]

        # Restricted evaluation of the features in singleCVIndex.
        for j in range(features / i):

            # Continue with for loop if already selected.
            if singleCVIndex[j] in selectedFeatureIndex:
                continue

            newFeature = shuffledData.iloc[:, singleCVIndex[j]]
            newFeature = newFeature.values.reshape((cases, 1))

            # Concatenate current features with previously selected features.
            expandedFeatures = np.column_stack((selectedFeatures, newFeature))

            score = p.apply_async(getCVScore, args=[labels, clf, expandedFeatures])
            indexScores.append((singleCVIndex[j], score.get()))

        # Again, sort (index, score) list, and select index that corresponds to
        # the best CV score.
        indexScores = sorted(indexScores, key=lambda (index, score):score, reverse=True)
        (bestIndex, bestScore) = indexScores[0]
        selectedFeatureIndex.append(bestIndex)
        selectedFeatureScores.append(bestScore)

    endTime = time.time()

    # Write results to a text file.
    filename = 'linear_all_results.txt'
    f = open(filename, 'w')
    for i in range(len(selectedFeatureScores)):
        index = selectedFeatureIndex[i]
        score = selectedFeatureScores[i]
        f.write('%i ; %10.8f' % (index, score))
        f.write("\n")

    f.write('Time taken: %f' % (endTime - startTime))
    f.write("\n")
    f.close()