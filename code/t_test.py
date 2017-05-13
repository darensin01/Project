import pandas as pd
import time
import multiprocessing
from sklearn import preprocessing
from scipy.stats import ttest_ind

# Constant terms.
STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'shuffledLabels'
DATA_KEY = 'shuffledData'
NUM_PROCESSES = 10

# Making data and labels global.
store = pd.HDFStore(STORE_NAME)
shuffledData = pd.read_hdf(STORE_NAME, DATA_KEY)
labels = (pd.read_hdf(STORE_NAME, LABEL_KEY)).values
store.close()

def getData():
    return pd.read_hdf(STORE_NAME, DATA_KEY)

def getLabels():
    return pd.read_hdf(STORE_NAME, LABEL_KEY)

def get_p_value(values):
    (cases, features) = shuffledData.shape

    # Find lists of **indices** where the labels are 0 or 1.
    index_label_1 = [j for j in xrange(cases) if labels[j] == 1]
    index_label_0 = [j for j in xrange(cases) if labels[j] == 0]

    p_values = []

    for i in values:
        feature = shuffledData.iloc[:, i]
        feature = preprocessing.scale(feature)
        values_label_1 = feature[index_label_1]
        values_label_0 = feature[index_label_0]

        t_statistic = ttest_ind(values_label_0, values_label_1, equal_var=False)[0]
        p_value = ttest_ind(values_label_0, values_label_1, equal_var=False)[1]
        p_values.append((i, t_statistic, p_value))

    return p_values

def partition(features):
    # Returns a list of list of indices that are partitioned according to
    # 'total', which is the number of processes.
    interval = features / NUM_PROCESSES
    return [range(index * interval, min((index + 1) * interval, features)) for index in range(NUM_PROCESSES + 1)]

if __name__ == "__main__":

    (cases, features) = shuffledData.shape

    # Find lists of **indices** where the labels are 0 or 1.
    index_label_1 = [j for j in xrange(cases) if labels[j] == 1]
    index_label_0 = [j for j in xrange(cases) if labels[j] == 0]

    start_time = time.time()
    pool = multiprocessing.Pool(processes = NUM_PROCESSES)
    results = pool.map(get_p_value, partition(features))

    # Flatten lists of list into a single list.
    p_values = [item for sublist in results for item in sublist]

    p_values = sorted(p_values, key=lambda(idx, t_stat, p_val):p_val, reverse=True)
    end_time = time.time()

    # Write results to a text file.
    filename = 't_test_results_unequal_var_t_stat.txt'
    f = open(filename, 'w')
    for (idx, t_stat, p_val) in p_values:
        f.write("(%i, %f, %f)" % (idx, t_stat, p_val))
        f.write("\n")

    f.write('Time taken: %f' % (end_time - start_time))
    f.write("\n")
    f.write('Number of processes: %i' % NUM_PROCESSES)
    f.write("\n")
    f.close()


