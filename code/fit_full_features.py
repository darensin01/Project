import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
import time
from sklearn import preprocessing

# Constant terms.
STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'shuffledLabels'
DATA_KEY = 'shuffledData'

if __name__ == "__main__":
    print "Reading from store..."

    # Read from data store.
    store = pd.HDFStore(STORE_NAME)
    shuffledData = pd.read_hdf(STORE_NAME, DATA_KEY)
    shuffledLabels = pd.read_hdf(STORE_NAME, LABEL_KEY)
    store.close()

    (cases, features) = shuffledData.shape
    labels = shuffledLabels.values

    clf = SVC(kernel='linear')
    shuffledData = shuffledData.apply(preprocessing.scale)
    start = time.time()
    cross_val_scores = cross_val_score(clf, shuffledData, y=labels, cv=5, scoring='accuracy')
    end = time.time()

    f = open("full_features_results.txt", 'w')
    f.write("Using linear...")
    f.write('%f' % np.mean(cross_val_scores))
    f.write("Time taken: %f" % (end - start))
    f.write("\n")

    clf = SVC(kernel='rbf')
    shuffledData = shuffledData.apply(preprocessing.scale)
    start = time.time()
    cross_val_scores = cross_val_score(clf, shuffledData, y=labels, cv=5, scoring='accuracy')
    end = time.time()

    f.write("Using rbf...")
    f.write('%f' % np.mean(cross_val_scores))
    f.write("Time taken: %f" % (end - start))
    f.write("\n")

    f.close()