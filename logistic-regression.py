from sklearn import neighbors
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

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

data = readData()
X = createDesignMatrix(data)
y = createLabelVector(data)
y = np.squeeze(np.asarray(y))
print(X.shape)


trainPoints, trainClasses, testPoints, testClasses = splitData7030(X, y)
# Over sample the minority classes
# NOTE: SHOULD OVERSAMPLE AFTER SPLITTING DATA NOT BEFORE
# OTHERWISE JUST LEARNING THE VALIDATION SET TOO

ratioDict = {3: 50, 4: 150, 5: 1063, 6: 1466, 7: 618, 8: 150, 9: 50 }

trainPointsOS, trainClassesOS = RandomOverSampler(random_state=0, ratio=ratioDict).fit_sample(trainPoints, trainClasses)
# points, classes = SMOTE(k_neighbors=3, ratio=0.7).fit_sample(X, y)
# points, classes = ADASYN(n_neighbors=4).fit_sample(X, y)


numFeatures = trainPointsOS.shape[1]
print(trainClassesOS)

# Show distribution of classes after over sampling
plt.hist([trainClasses, trainClassesOS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Oversampling', 'Oversampling'])
plt.legend()
plt.show()


trainPointsOS = featureNormalize(trainPointsOS)

# if TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD":
#     acc = []
#     numDataSplits = TRAINING_PARAMS['NUM_SPLITS']
#     print("Cross Validation used for splitting")
#     for i in range(numDataSplits):
#         trainPoints, trainClasses, testPoints, testClasses = splitUpDataCrossVal(points, classes, numDataSplits, crossValIndex=i)
#         currentAcc, yPrediction = trainModel(trainPoints, trainClasses, testPoints, testClasses)
#         acc.append(currentAcc)
#     averageAcc = np.mean(acc)
# else:
#     print("70/30 method used for splitting")
#     trainPoints, trainClasses, testPoints, testClasses = splitData7030(points, classes)
#     averageAcc, yPrediction = trainModel(trainPoints, trainClasses, testPoints, testClasses)

averageAcc, yPrediction = trainModel(trainPointsOS, trainClassesOS, testPoints, testClasses)

print(averageAcc)
plt.hist([testClasses, trainClassesOS, yPrediction], bins=range(3, 11), align='left', rwidth=0.5,
         label=["actual", "over sampled", "prediction"])
plt.legend(loc='upper right')
plt.show()
