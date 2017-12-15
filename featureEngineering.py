
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.utils import to_categorical

from collections import Counter
from sklearn import linear_model
import numpy as np

import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

from util import (
    readData,
    createDesignMatrix,
    createLabelVector,
    printM,
    splitData7030,
    splitUpDataCrossVal,
    featureNormalize,
    multiclassToBinaryClass,
    ACTIVE_DATASET,
    TRAINING_PARAMS
)

from plotCallback import (PlotLossAccuracy)

NUM_CLASSES = 10


# In[2]:



data = readData()
X = createDesignMatrix(data)
X = featureNormalize(X)

y = createLabelVector(data)
y = np.squeeze(np.asarray(y))
print("X: ", X.shape)
print("y: ", y.shape)

# np.savetxt("split.csv", yTrain)

data = readData(filename='testData.csv')
xTest = createDesignMatrix(data)
xTest = featureNormalize(xTest)

yTest = createLabelVector(data)
yTest = np.squeeze(np.asarray(yTest))
# yTest = to_categorical(yTest, num_classes=10)


# In[3]:


# print(np.corrcoef(X[:,0], y)[0,1])
# ex,log(x),x2,x3,tanh(x)}x′∈{ex,log⁡(x),x2,x3,tanh⁡(x)}

def corelationCoefficients(X, y):
    r = []
    for i in range(X.shape[1]):
        tmp = X[:, i]
        cc = np.corrcoef(tmp, y)[0, 1]
        r.append(cc)
    return r
    
# Volatile Acidity
# Chlorides
# Density
# Alcohol


def getGoodFeatures(cc):
    goodFeatures = dict()
    for (count, val) in enumerate(cc):
        if(abs(val) > 0.2):
            goodFeatures[count] = val
    return goodFeatures


# In[8]:


features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
            "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
x = range(X.shape[1])


font = {'family' : 'normal',
        'size'   : 22}

plt.rc('font', **font)


linearCC = corelationCoefficients(X, y)
plt.scatter(x, linearCC, label="Linear")
print("Linear: ", getGoodFeatures(linearCC))

squareCC = corelationCoefficients(np.power(X, 2), y)
plt.scatter(x, squareCC, label="Square")
print("Square: ", getGoodFeatures(squareCC))


expCC = corelationCoefficients(np.exp(X), y)
plt.scatter(x, expCC, label="Exp")
print("Exponentional: ", getGoodFeatures(expCC))

# x = range(X.shape[1])
# plt.xticks(x, features)


plt.legend()
plt.xlabel("Feature")
plt.ylabel("Correlation Coefficient")
plt.show()

