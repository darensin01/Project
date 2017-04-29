import pandas as pd
import numpy as np
import time
import math
import multiprocessing
from joblib import Parallel, delayed
from sklearn import preprocessing

STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'shuffledLabels'
DATA_KEY = 'shuffledData'
K = 100

def compute_rqn(feature, index_label_0, index_label_1, prob_label_1):
    prob_label_0 = 1 - prob_label_1

    # E(X ^ 2).
    exp_feature_sq = np.mean(np.square(feature))

    # E(X | Y = y).
    exp_feature_label_1 = np.mean(feature[index_label_1])
    exp_feature_label_0 = np.mean(feature[index_label_0])

    # var(X). Returns unbiased estimate of variance with ddof = 1.
    var_feature = np.var(feature, ddof=1)

    # E(var(X | Y))
    cond_exp = exp_feature_sq - (prob_label_1 * exp_feature_label_1 ** 2) \
               - (prob_label_0 * exp_feature_label_0 ** 2)

    # Compute correlation coefficient.
    return 1 - cond_exp / var_feature

def compute_sc(f, q):
    numerator = np.dot(f.T, q) ** 2
    denominator = np.dot(f.T, f) * np.dot(q.T, q)
    return numerator / denominator

def get_feature(idx, data):
    feature = data.iloc[:, idx]
    return preprocessing.scale(feature)

def compute_mgs(qs):
    (m, n) = qs.shape
    Q = np.zeros((m, n))

    for k in range(n):
        Q[:, [k]] = qs[:, k] / np.linalg.norm(qs[:, k])

        for j in range(k + 1, n):
            qk = Q[:, k]
            aj = qs[:, j]
            r = np.dot(qk, aj)[0, 0]
            qs[:, [j]] -= (qk * r).reshape((m, 1))

    print "\n"
    print "Checking for small dot products..."
    for i in range(n-1):
        print np.dot(Q[:, i], Q[:, i+1])

    return Q

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

    # List of orthogonal vectors.
    orth_q = []

    # List of selected feature indices.
    selected_idx = []

    start_time = time.time()

    # Compute correlation coefficient (rqn) between each feature and target
    # labels.
    for i in range(features):
        feature = get_feature(i, shuffledData)

        rqn = compute_rqn(feature, index_label_0, index_label_1, prob_label_1)
        rqns.append(rqn)

    # Return list of sorted tuples (index, rqn score).
    index_rqn = sorted(enumerate(rqns), key=lambda(idx, val):val, reverse=True)

    print "\n"
    print "Chosen index is %i" % index_rqn[0][0]

    best_index = index_rqn[0][0]
    orth_q = (np.matrix(get_feature(best_index, shuffledData))).T
    selected_idx.append(best_index)

    for h in range(2, K+1):

        # Creates a list of candidate indices
        candidate_idx = [x for x in range(features) if x not in selected_idx]

        # Keep a list of values of monitoring criterion.
        criterion = []
        for i in candidate_idx:
            feature = get_feature(i, shuffledData)

            # Use previously computed list.
            rqn = [score for idx, score in index_rqn if idx == i][0]

            # Calculate multiple correlation coefficient between each feature,
            # and the orthogonalised features.
            sc = 0
            for k in range(h - 1):
                sc += compute_sc(feature, orth_q[:, k])

            criterion.append((i, rqn - sc))

        index_criterion = sorted(criterion, key=lambda (idx, val): val, reverse=True)

        best_index = index_criterion[0][0]
        print "Chosen index is %i" % best_index
        selected_idx.append(best_index)

        # Perform MGS on orthogonalised q vectors, to obtain orthogonalised
        # vectors.
        orth_q = np.column_stack((orth_q, get_feature(best_index, shuffledData)))
        orth_q = compute_mgs(np.matrix(orth_q))

    end_time = time.time()

    # Write results to a text file.
    filename = 'mrmmc_results_100.txt'
    f = open(filename, 'w')
    for i in range(len(selected_idx)):
        f.write("%i" % (selected_idx[i]))
        f.write("\n")

    f.write('Time taken: %f' % (end_time - start_time))
    f.write("\n")
    f.close()



