import pandas as pd
import time
import multiprocessing
from sklearn import preprocessing
from scipy.stats import ttest_ind
from joblib import Parallel, delayed

# Constant terms.
STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'shuffledLabels'
DATA_KEY = 'shuffledData'
NUM_PROCESSES = multiprocessing.cpu_count()

def getData():
    return pd.read_hdf(STORE_NAME, DATA_KEY)

def getLabels():
    return pd.read_hdf(STORE_NAME, LABEL_KEY)

def get_p_value(idx, shuffledData, labels):
    (cases, features) = shuffledData.shape

    interval = features / NUM_PROCESSES

    # Find lists of **indices** where the labels are 0 or 1.
    index_label_1 = [j for j in xrange(cases) if labels[j] == 1]
    index_label_0 = [j for j in xrange(cases) if labels[j] == 0]

    p_values = []

    for i in range(idx * interval, min(features, (idx + 1) * interval)):
        feature = shuffledData.iloc[:, i]
        feature = preprocessing.scale(feature)
        values_label_1 = feature[index_label_1]
        values_label_0 = feature[index_label_0]

        #t_statistic = ttest_ind(values_label_0, values_label_1)[0]
        p_value = ttest_ind(values_label_0, values_label_1)[1]
        p_values.append((i, abs(p_value)))
        #print i

    return p_values

if __name__ == "__main__":

    store = pd.HDFStore(STORE_NAME)

    shuffledData = getData()
    shuffledLabels = getLabels()
    labels = shuffledLabels.values

    (cases, features) = shuffledData.shape

    # Find lists of **indices** where the labels are 0 or 1.
    index_label_1 = [j for j in xrange(cases) if labels[j] == 1]
    index_label_0 = [j for j in xrange(cases) if labels[j] == 0]

    start_time = time.time()
    #results = Parallel(n_jobs=NUM_PROCESSES)(delayed(get_p_value)(i, shuffledData, labels) for i in xrange(NUM_PROCESSES + 1))

    # Flatten lists of list into a single list.
    #p_values = [item for sublist in results for item in sublist]

    p_values = []

    for i in range(features):
        feature = shuffledData.iloc[:, i]
        feature = preprocessing.scale(feature)
        values_label_1 = feature[index_label_1]
        values_label_0 = feature[index_label_0]

        #t_statistic = ttest_ind(values_label_0, values_label_1)[0]
        p_value = ttest_ind(values_label_0, values_label_1)[1]
        p_values.append((i, abs(p_value)))

    p_values = sorted(p_values, key=lambda(idx, p_val):p_val, reverse=True)
    end_time = time.time()

    #print p_values

    store.close()

    # Write results to a text file.
    filename = 't_test_results_no_parallel.txt'
    f = open(filename, 'w')
    for (idx, p_val) in p_values:
        f.write("(%i, %f)" % (idx, p_val))
        f.write("\n")

    f.write('Time taken: %f' % (end_time - start_time))
    f.write("\n")
    f.write('Number of processes: %i' % NUM_PROCESSES)
    f.write("\n")
    f.close()


