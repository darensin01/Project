import pandas as pd
import numpy as np
import math
import time
import multiprocessing
from joblib import Parallel, delayed
from sklearn import preprocessing

STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'shuffledLabels'
DATA_KEY = 'shuffledData'

def get_rqn(i, index_label_1, index_label_0, prob_label_1, shuffledData):

    # Get probability of label being 0.
    prob_label_0 = 1 - prob_label_1

    # Get feature based on index.
    feature = shuffledData.iloc[:, i]

    # E(X ^ 2).
    exp_feature_sq = np.mean(np.square(feature.values))

    # E(X | Y = y).
    exp_feature_label_1 = np.mean(feature.iloc[index_label_1])
    exp_feature_label_0 = np.mean(feature.iloc[index_label_0])

    # var(X). Returns unbiased estimate of variance with ddof = 1.
    var_feature = np.var(feature, ddof=1)

    # E(var(X | Y))
    cond_exp = exp_feature_sq - (prob_label_1 * exp_feature_label_1 ** 2) \
                              - (prob_label_0 * exp_feature_label_0 ** 2)

    # Compute correlation coefficient.
    rqn = math.sqrt(1 - cond_exp / var_feature)

    # Return also the index.
    return i, rqn

if __name__ == '__main__':

    # Get data from store.
    store = pd.HDFStore(STORE_NAME)
    shuffledData = pd.read_hdf(store, DATA_KEY)
    shuffledLabels = pd.read_hdf(store, LABEL_KEY)
    store.close()

    (cases, features) = shuffledData.shape
    labels = shuffledLabels.values

    # Calculate probability of P(Y=1) and P(Y=0).
    prob_label_1 = np.mean(labels)
    prob_label_0 = 1 - prob_label_1

    # Find lists of **indices** where the labels are 0 or 1.
    index_label_1 = [j for j in xrange(cases) if labels[j] == 1]
    index_label_0 = [j for j in xrange(cases) if labels[j] == 0]

    # List of correlation coefficients (rqn).
    rqns = []

    #start_time = time.time()
    #num_cores = multiprocessing.cpu_count()
    #print "number of cores is %i" % num_cores
    #rqns = Parallel(n_jobs=num_cores)(delayed(get_rqn)(i, index_label_1, index_label_0, prob_label_1, shuffledData) for i in range(features))
    #end_time = time.time()

    start_time = time.time()

    # Compute correlation coefficient between each feature and target labels.
    for i in range(1):
        feature = shuffledData.iloc[:, i]

        # E(X ^ 2).
        exp_feature_sq = np.mean(np.square(feature.values))

        # E(X | Y = y).
        exp_feature_label_1 = np.mean(feature.iloc[index_label_1])
        exp_feature_label_0 = np.mean(feature.iloc[index_label_0])

        # var(X). Returns unbiased estimate of variance with ddof = 1.
        var_feature = np.var(feature, ddof=1)

        # E(var(X | Y))
        cond_exp = exp_feature_sq - (prob_label_1 * exp_feature_label_1 ** 2) \
                   - (prob_label_0 * exp_feature_label_0 ** 2)

        # Compute correlation coefficient.
        rqn = math.sqrt(1 - cond_exp / var_feature)
        rqns.append(rqn)
        print rqn

    end_time = time.time()

    # Return list of sorted tuples (index, rqn score).
    index_rqn = sorted(enumerate(rqns), key=lambda(idx, val):val, reverse=True)
    #index_rqn = sorted(rqns, key=lambda (idx, val): val, reverse=True)

    # Return list of **indices** of sorted rqn scores.
    #sorted_indices = map(lambda(idx, val):idx, index_rqn)

    # Write results to a text file.
    filename = 'rqn_scores.txt'
    f = open(filename, 'w')
    for tup in index_rqn:
        index = tup[0]
        score = tup[1]
        f.write('(%i, %10.8f)' % (index, score))
        f.write("\n")

    f.write('Time taken: %f' % (end_time - start_time))
    f.write("\n")
    f.close()




