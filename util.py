# Silence Tensor Flowing optimization warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import numpy as np
import matplotlib.pyplot as plt
from pandas import *
from sklearn.preprocessing import scale


from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

from collections import Counter

# Load data from config file (see config.py)
from config import TRAINING_PARAMS, ACTIVE_DATASET, SHOW_PLOTS


# Prints Matrices in a nicer way
def printM(dataset):
    print(DataFrame(dataset), "\n")


# Read's data from CSV File
def readData(filename=ACTIVE_DATASET['FILE_NAME']):
    csvData = np.recfromcsv(filename, delimiter=ACTIVE_DATASET['DELIMETER'], filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ') 
    
    if(TRAINING_PARAMS['DESIRED_NUM_INSTANCES']):
        numInstances = min(TRAINING_PARAMS['DESIRED_NUM_INSTANCES'], len(csvData))
    else:
        numInstances = len(csvData)
    return csvData[:numInstances]


# Finds the column numbers of the features to be included in the design matrix X
def getDesiredFeatureIndices(allFeatureNames, features, omitFeatures):
    return [i for i, x in enumerate(allFeatureNames) if ((x in features and not omitFeatures) or (x not in features and omitFeatures))]


# Builds the design matrix X inlcuding a column vector of 1s for the bias terms
def createDesignMatrix(dataset):
    if(ACTIVE_DATASET['OMIT_FEATURES']):
        numDesignMatrixFeatures = len(dataset[0]) - len(ACTIVE_DATASET['FEATURES'])
    else:
        numDesignMatrixFeatures = len(ACTIVE_DATASET['FEATURES'])
    X = np.ones((len(dataset), numDesignMatrixFeatures))
    featureIndices = getDesiredFeatureIndices(list(dataset.dtype.names), ACTIVE_DATASET['FEATURES'], ACTIVE_DATASET['OMIT_FEATURES'])
    currentCol = 0
    for i, row in enumerate(dataset):
        for j, col in enumerate(row):
            if(j in featureIndices):
                X[i,currentCol] = col
                currentCol = currentCol + 1
        currentCol = 0
    return X


# Creates the specified label (output) vector from the dataset
def createLabelVector(dataset):
    y = np.transpose(np.matrix(dataset[ACTIVE_DATASET['LABEL']]))
    return y

# Splits data according using a 70/30 split
def splitData7030(X, y):
    numTrainInstances = round(len(X)*0.7)
    xTrain = X[:numTrainInstances, :]
    yTrain = y[:numTrainInstances]
    xTest = X[numTrainInstances:]
    yTest = y[numTrainInstances:]
    return xTrain, yTrain, xTest, yTest

# Splits up data according to the number of folds, and the current fold index
def splitUpDataCrossVal(X, y, splitFactor, crossValIndex=0):
    testSetSize = round(X.shape[0] / splitFactor)
    startTestIndex =  crossValIndex * testSetSize

    xTest = X[startTestIndex:startTestIndex+testSetSize]
    yTest = y[startTestIndex:startTestIndex+testSetSize]

    xTrain = X[:startTestIndex, :]
    xTrain = np.append(xTrain, np.array(X[startTestIndex+testSetSize:]), axis=0)

    yTrain = y[:startTestIndex]
    yTrain = np.append(yTrain, np.array(y[startTestIndex+testSetSize:]), axis=0)
    return xTrain, yTrain, xTest, yTest



# Normalizes features to Standard Normal Variable or maps them over [0,1] range
def featureNormalize(dataset):
    if(TRAINING_PARAMS['NORMALIZE_METHOD'] == "MINMAX"):
        print("Using min-max normalization")
        mins = np.amin(dataset, axis=0)
        maxs = np.amax(dataset, axis=0)
        return (dataset - mins) / (maxs-mins)
    else: 
        mu = np.mean(dataset,axis=0)
        sigma = np.std(dataset,axis=0)
        print("Using Z-Score Normalization")
        return (dataset-mu)/sigma

# Creates a binary classification from a multi classification
def multiclassToBinaryClass(labelVector, threshold):
    for i, label in enumerate(labelVector):
        if (label <= threshold):
            labelVector[i][0] = 0
        else:
            labelVector[i][0] = 1
    return labelVector

# Over sample the minority classes
# NOTE: SHOULD OVERSAMPLE AFTER SPLITTING DATA NOT BEFORE
# OTHERWISE JUST LEARNING THE VALIDATION SET TOO
def splitAndOverSample(X, y, numDataSplits=None, crossValIndex=None):

    # Split the data
    if numDataSplits:
      xTrain, yTrain, xVal, yVal = splitUpDataCrossVal(X, y, numDataSplits, crossValIndex)
    else:
      xTrain, yTrain, xVal, yVal = splitData7030(X, y)

    # Sample the Data
    if TRAINING_PARAMS['OVER_SAMPLE']:
        counts = Counter(yTrain)
        if(TRAINING_PARAMS['USE_OS_DICT']):
            print("Using OS Dict")
            # Define oversampling amounts
            ratioDict = {3: max(50, counts[3]), 4: max(200, counts[4]), 5: counts[5],
                            6: counts[6], 7: counts[7], 8: max(200, counts[8]), 9: max(50, counts[9])}

            # Oversample the training data
            xTrainOS, yTrainOS = RandomOverSampler(random_state=0, ratio=ratioDict).fit_sample(xTrain, yTrain)
            print('Oversampling')
            print('Rating distribution: ', sorted(Counter(yTrainOS).items()))
            

            if SHOW_PLOTS:
                # Show distribution of classes after over sampling
                plt.hist([yTrain, yTrainOS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Oversampling', 'Oversampling'])
                plt.legend()
                plt.show()

            return xTrainOS, yTrainOS, xVal, yVal
        
        else:
            
            print("Not using OS Dict")

            # Oversample the training data
            xTrainOS, yTrainOS = RandomOverSampler(random_state=0).fit_sample(xTrain, yTrain)
            print('Oversampling')
            print('Rating distribution: ', sorted(Counter(yTrainOS).items()))
            

            if SHOW_PLOTS:
                # Show distribution of classes after over sampling
                plt.hist([yTrain, yTrainOS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Oversampling', 'Oversampling'])
                plt.legend()
                plt.show()

            return xTrainOS, yTrainOS, xVal, yVal
    
    # No Oversampling
    else:
      return xTrain, yTrain, xVal, yVal
