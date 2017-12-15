
# coding: utf-8

# In[21]:


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


# In[22]:


# Over sample the minority classes
# NOTE: SHOULD OVERSAMPLE AFTER SPLITTING DATA NOT BEFORE
# OTHERWISE JUST LEARNING THE VALIDATION SET TOO
def splitAndOverSample(X, y, numDataSplits=None, crossValIndex=None):

    # Split the data
    if numDataSplits:
        xTrain, yTrain, xVal, yVal = splitUpDataCrossVal(X, y, numDataSplits, crossValIndex)
    else:
        xTrain, yTrain, xVal, yVal = splitData7030(X, y)

    # If no sampling to be done
    if not TRAINING_PARAMS['BALANCE_SAMPLING']:

        ## Binary encode the classes data
        yTrain = to_categorical(yTrain, num_classes=NUM_CLASSES)
        yVal = to_categorical(yVal, num_classes=NUM_CLASSES)
        print("After binary encoding y: ", yTrain.shape)
        return  xTrain, yTrain, xVal, yVal
    else:
        counts = Counter(yTrain)

        if TRAINING_PARAMS['BALANCE_SAMPLING'] == 'OVER':

            # Define oversampling amounts
            ratioDict = {3: max(20, counts[3]), 4: max(80, counts[4]), 5: counts[5],
                         6: counts[6], 7: counts[7], 8: max(80, counts[8]), 9: max(20, counts[9])}

            # Oversample the training data
            xTrainOS, yTrainOS = RandomOverSampler(random_state=0, ratio=ratioDict).fit_sample(xTrain, yTrain)
            # xTrainOS, yTrainOS = SMOTE(k_neighbors=3, ratio=ratioDict).fit_sample(xTrain, yTrain)
            # xTrainOS, yTrainOS = ADASYN(n_neighbors=4, ratio=ratioDict).fit_sample(xTrain, yTrain)

            print('Oversampling')
            print('Rating distribution: ', sorted(Counter(yTrainOS).items()))

            # Show distribution of classes after over sampling
            plt.hist([yTrain, yTrainOS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Oversampling', 'Oversampling'])
            plt.legend()
            plt.show()

            # Binary encode
            yTrainOS = to_categorical(yTrainOS, num_classes=10)
            yVal = to_categorical(yVal, num_classes=10)
            print("After binary encoding y: ", yTrainOS.shape)

            return xTrainOS, yTrainOS, xVal, yVal
        else:
            ratioDict = {3: counts[3], 4: counts[4], 5: min(counts[5], 400),
                         6: min(counts[6], 800), 7: min(counts[7], 400), 8: counts[8], 9: counts[9]}
            xTrainUS, yTrainUS = RandomUnderSampler(random_state=0, ratio=ratioDict).fit_sample(xTrain, yTrain)

            print('Using Undersampling')
            print('Category distribution: ', sorted(Counter(yTrainUS).items()))

            # Show distribution of classes after under sampling
            plt.hist([yTrain, yTrainUS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Undersampling', 'Undersampling'])
            plt.legend()
            plt.show()

            # Binary encode
            yTrainUS = to_categorical(yTrainUS, num_classes=10)
            yVal = to_categorical(yVal, num_classes=10)
            print("After binary encoding y: ", yTrainUS.shape)

            return xTrainUS, yTrainUS, xVal, yVal


# In[23]:



data = readData()
X = createDesignMatrix(data)
X = featureNormalize(X)

y = createLabelVector(data)
y = np.squeeze(np.asarray(y))
print("X: ", X.shape)
print("y: ", y.shape)


print(X)
print(y)
# np.savetxt("split.csv", yTrain)

data = readData(filename='testData.csv')
xTest = createDesignMatrix(data)
xTest = featureNormalize(xTest)

yTest = createLabelVector(data)
yTest = np.squeeze(np.asarray(yTest))
yTest = to_categorical(yTest, num_classes=10)


# In[24]:





# Define the network
model = Sequential()

model.add(Dense(256, activation='relu', input_dim=X.shape[1]))
model.add(Dropout(0.3))

# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))


# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(NUM_CLASSES, activation='softmax'))

# model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
pltCallBack = PlotLossAccuracy()


# In[25]:




accuracies = []
# for i in range(10):
# xTrain, yTrain, xVal, yVal = splitAndOverSample(X, y, 10, i)
xTrain, yTrain, xVal, yVal = splitAndOverSample(X, y)
print("xTrain: ", xTrain.shape)
print("yTrain: ", yTrain.shape)
print("xVal: ", xVal.shape)
print("yVal: ", yVal.shape)

# Train the model
model.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=150, callbacks=[pltCallBack])
pltCallBack.show_plots()

# Predict the validation set
predictions = model.predict(xVal)
# print("Predicter: ")
# print(predictions[0], "\n")

# Extract the predicted categories
predictionCat = np.argmax(predictions, axis=1)
print(predictionCat)

# Plot histogram of predicted category distribution
plt.hist(predictionCat, bins=range(3, 11), align='left', rwidth=0.5)
plt.show()

# print("Actual: ")
# print(yVal[0], "\n")

# evaluate the model
scores = model.evaluate(xVal, yVal)
print(": Validation Accuracy = ", scores[1])
testScores = model.evaluate(xTest, yTest)
print(": Test Accuracy = ", testScores[1])


# In[26]:


testScores = model.evaluate(xTest, yTest)
print("Accuracy on Test set = ", testScores[1])

