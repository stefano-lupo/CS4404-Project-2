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


def trainModel(xtrain, ytrain, xtest, ytest):
    ## interesting options:
    # class_weight="balanced",
    # multi_class="multinomial", solver=‘newton-cg’ or ‘sag’ or ‘saga’ or ‘lbfgs’
    # c=<some_float> higher for less regularization
    logreg = linear_model.LogisticRegression(multi_class="multinomial", solver="lbfgs")
    logreg.fit(xtrain, ytrain)

    # Predict the test values
    yPrediction = logreg.predict(xtest)

    # Get the output probabilities of each class for each test instance
    probabilities = logreg.predict_proba(xtest)
    print(probabilities.shape)

    # Get the accuracy of the predictions
    accuracy = logreg.score(xtest, ytest)
    return accuracy, yPrediction


def getNumSamplesFromPercent(n, percentages):
    numSamples = dict()
    for i, val in enumerate(percentages):
        numSamples[i] = n * percentages[i]

    print(numSamples)
    return numSamples


# Over sample the minority classes
# NOTE: SHOULD OVERSAMPLE AFTER SPLITTING DATA NOT BEFORE
# OTHERWISE JUST LEARNING THE VALIDATION SET TOO
def splitAndOverSample(X, y, numDataSplits=None, crossValIndex=None):

    # Split the data
    if numDataSplits:
        trainPoints, trainClasses, testPoints, testClasses = splitUpDataCrossVal(X, y, numDataSplits, crossValIndex)
    else:
        trainPoints, trainClasses, testPoints, testClasses = splitData7030(X, y)

    if not TRAINING_PARAMS['BALANCE_SAMPLING']:
        return trainPoints, trainClasses, testPoints, testClasses
    else:
        counts = Counter(trainClasses)

        if TRAINING_PARAMS['BALANCE_SAMPLING'] == 'OVER':

            # Define oversampling amounts
            ratioDict = {3: max(20, counts[3]), 4: max(150, counts[4]), 5: counts[5],
                         6: counts[6], 7: counts[7], 8: max(150, counts[8]), 9: max(20, counts[9])}

            # Oversample the training data
            trainPointsOS, trainClassesOS = RandomOverSampler(random_state=0, ratio=ratioDict).fit_sample(trainPoints, trainClasses)
            # trainPointsOS, trainClassesOS = SMOTE(k_neighbors=3, ratio=ratioDict).fit_sample(trainPoints, trainClasses)
            # trainPointsOS, trainClassesOS = ADASYN(n_neighbors=4, ratio=ratioDict).fit_sample(trainPoints, trainClasses)

            print(sorted(Counter(trainClassesOS).items()))

            # Show distribution of classes after over sampling
            # plt.hist([trainClasses, trainClassesOS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Oversampling', 'Oversampling'])
            # plt.legend()
            # plt.show()

            return trainPointsOS, trainClassesOS, testPoints, testClasses
        else:
            ratioDict = {3: counts[3], 4: counts[4], 5: min(counts[5], 400),
                         6: min(counts[6], 800), 7: min(counts[7], 400), 8: counts[8], 9: counts[9]}
            trainPointsUS, trainClassesUS = RandomUnderSampler(random_state=0, ratio=ratioDict).fit_sample(trainPoints, trainClasses)

            print(sorted(Counter(trainClassesUS).items()))

            # Show distribution of classes after under sampling
            # plt.hist([trainClasses, trainClassesUS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Undersampling', 'Undersampling'])
            # plt.legend()
            # plt.show()

            return trainPointsUS, trainClassesUS, testPoints, testClasses


data = readData()
X = createDesignMatrix(data)
X = featureNormalize(X)

y = createLabelVector(data)
y = np.squeeze(np.asarray(y))
print(X.shape)


if TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD":
    acc = []
    numDataSplits = TRAINING_PARAMS['NUM_SPLITS']
    print("Cross Validation used for splitting")
    for i in range(numDataSplits):

        # Split and oversample data
        trainPoints, trainClasses, testPoints, testClasses \
            = splitAndOverSample(X, y, numDataSplits=numDataSplits, crossValIndex=i)

        # Train the model
        currentAcc, yPrediction = trainModel(trainPoints, trainClasses, testPoints, testClasses)
        acc.append(currentAcc)

    averageAcc = np.mean(acc)

else:
    print("70/30 method used for splitting")
    trainPoints, trainClasses, testPoints, testClasses = splitAndOverSample(X, y)
    averageAcc, yPrediction = trainModel(trainPoints, trainClasses, testPoints, testClasses)

print(averageAcc)
plt.hist([testClasses, trainClasses, yPrediction], bins=range(3, 11), align='left', rwidth=0.5,
         label=["actual", "sampled", "prediction"])
plt.legend(loc='upper right')
plt.show()
