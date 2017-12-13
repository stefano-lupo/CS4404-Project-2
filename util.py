import numpy as np
import matplotlib.pyplot as plt
from pandas import *
from sklearn.preprocessing import scale

# Load data from config file (see config.py)
from config import TRAINING_PARAMS, ACTIVE_DATASET


# Prints Matrices in a nicer way
def printM(dataset):
    print(DataFrame(dataset), "\n")


# Read's data from CSV File
def readData():
    csvData = np.recfromcsv(ACTIVE_DATASET['FILE_NAME'], delimiter=ACTIVE_DATASET['DELIMETER'], filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ') 
    if(TRAINING_PARAMS['DESIRED_NUM_INSTANCES']):
        numInstances = min(TRAINING_PARAMS['DESIRED_NUM_INSTANCES'], len(csvData))
    else:
        numInstances = len(csvData)
    return csvData[:numInstances]

def changeToTestData():
    ACTIVE_DATASET['FILE_NAME'] = "testData.csv"


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
