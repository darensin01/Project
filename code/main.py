import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import numpy as np

# Constant terms.
STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'label'
DATA_KEY = 'data'

# Reads from Excel file the case/control labels. Counts number of case and
# control individuals. Creates a series (0-indexed) of the case/controls.
def createDiseaseLabel():

    status = pd.read_excel('disease_status.xlsx',
                           sheetname='Sheet1',
                           header=None)

    # Append the content of the excel file into a Series.
    srs = pd.Series(dtype=int)

    # Count number of case and control (checking purposes).
    status_one_count = 0
    status_two_count = 0

    # Iterate through disease_status text file.
    for index, row in status.iterrows():
        if row[0] == "disease_status: 1":
            # Control
            status_one_count += 1
            srs = srs.append(pd.Series([0], index=[index]))
        else:
            # Case
            status_two_count += 1
            srs = srs.append(pd.Series([1], index=[index]))

    #print status_one_count, status_two_count
    #print status_one_count + status_two_count
    return srs

def getData():
    print "Getting data matrix..."
    return pd.read_hdf(STORE_NAME, DATA_KEY)

def getLabels():
    print "Getting labels..."
    return pd.read_hdf(STORE_NAME, LABEL_KEY)

def trainWithRandomForest(data, labels):
    print "Training RF..."
    rf = RandomForestClassifier()
    return getCVScore(rf, data, labels)

# Arbitrary L1 ratio (which determines how much L1/L2 penalty to impose).
# Can consider conducting grid search on L1 ratio.
def trainWithENet(data, labels):    
    print "Training Elastic net..."
    eNet = ElasticNetCV(l1_ratio=0.6)
    return getCVScore(eNet, data, labels)

# Default C parameter: 1.0.
# Default kernel: RBF kernel.
# Default gamma for RBF kernel: 1/n_features.
def trainWithSVM(data, labels):    
    print "Training SVM..."
    clf = SVC()
    return getCVScore(clf, data, labels)

# Default number of folds for cross validation: 3.
def getCVScore(classifier, data, labels):
    return cross_val_score(classifier, data, y=labels, n_jobs=-1, cv=5)

# Main method

if __name__ == "__main__":
    # Data is read from csv file, and the resulting data frame is converted into a
    #  HDFStore for fast retrieval. Don't have to read csv file each time we run our
    # algorithm. Data in the store is indexed by 'data'.
    print "Getting store..."
    store = pd.HDFStore(STORE_NAME)
    print "Done getting store!\n"

    # Returns a series representing the labels (needed for supervised learning).
    labels = getLabels()

    # Get the data matrix. Need to transpose the dataframe to make the dimensions
    # compatible with the labels vector.
    data = getData()

    # df = pd.read_csv('first100Rows.csv', header=0, index_col=0)
    # df.to_hdf(store, 'dataTop100')
    # data = df.T  # Only 100 features
    # print data.shape

    store.close()

    # Shuffle the data and the labels, so that the bias associated with the training
    # would be alleviated. Set random_state=0 for reproducibility.
    print "Shuffling data and labels...\n"
    dataShuffled, labelsShuffled = shuffle(data.T, labels, random_state=0)

    # eNetCVScore = trainWithENet(dataShuffled, labelsShuffled)
    # rfCVScore = trainWithRandomForest(dataShuffled, labelsShuffled)
    svmCVScore = trainWithSVM(dataShuffled, labelsShuffled)

    f = open("results.txt", 'w')
    f.write("SVM cross validation score:\n")
    f.write(svmCVScore)
    f.close()




