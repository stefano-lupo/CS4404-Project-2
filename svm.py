from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV

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


NUM_CLASSES = 10
showPlots = False

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
    if TRAINING_PARAMS['BALANCE_SAMPLING'] == 'OVER':
      counts = Counter(yTrain)
      # Define oversampling amounts
      ratioDict = {3: max(200, counts[3]), 4: max(400, counts[4]), 5: counts[5],
                    6: counts[6], 7: counts[7], 8: max(400, counts[8]), 9: max(100, counts[9])}

      # Oversample the training data
      xTrainOS, yTrainOS = RandomOverSampler(random_state=0, ratio=ratioDict).fit_sample(xTrain, yTrain)
      # xTrainOS, yTrainOS = SMOTE(k_neighbors=3, ratio=ratioDict).fit_sample(xTrain, yTrain)
      # xTrainOS, yTrainOS = ADASYN(n_neighbors=4, ratio=ratioDict).fit_sample(xTrain, yTrain)

      print('Oversampling')
      print('Rating distribution: ', sorted(Counter(yTrainOS).items()))
      

      if showPlots:
        # Show distribution of classes after over sampling
        plt.hist([yTrain, yTrainOS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Oversampling', 'Oversampling'])
        plt.legend()
        plt.show()

      return xTrainOS, yTrainOS, xVal, yVal


    elif TRAINING_PARAMS['BALANCE_SAMPLING'] == 'OVER':
      counts = Counter(yTrain)
      ratioDict = {3: counts[3], 4: counts[4], 5: min(counts[5], 400),
                    6: min(counts[6], 800), 7: min(counts[7], 400), 8: counts[8], 9: counts[9]}
      xTrainUS, yTrainUS = RandomUnderSampler(random_state=0, ratio=ratioDict).fit_sample(xTrain, yTrain)

      print('Using Undersampling')
      print('Category distribution: ', sorted(Counter(yTrainUS).items()))

      if showPlots:
        # Show distribution of classes after under sampling
        plt.hist([yTrain, yTrainUS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Undersampling', 'Undersampling'])
        plt.legend()
        plt.show()

      return xTrainUS, yTrainUS, xVal, yVal

    else:
      return xTrain, yTrain, xVal, yVal

data = readData()
X = createDesignMatrix(data)
X = featureNormalize(X)

y = createLabelVector(data)
y = np.squeeze(np.asarray(y))

# thresholder = lambda x : 1 if x>7 else 0
y = np.array([1 if x>7 else 0 for x in y]);
print("X: ", X.shape)
print("y: ", y.shape)
print(y)

## 1st Option
# clf = svm.SVC()
# scores = cross_val_score(clf, X, y, cv=10)
# print(scores)
# print("Accuracy: (95% confidence) %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

## 2nd Option
## Split data
# xTrain, xVal, yTrain, yVal = train_test_split(X, y, test_size=0.3, random_state=0)
# scaler = preprocessing.StandardScaler().fit(xTrain)
# xTrainTransformed = scaler.transform(xTrain)
# clf = svm.SVC(C=1).fit(xTrainTransformed, yTrain)
# xValTransformed = scaler.transform(xVal)
# scores = clf.score(xValTransformed, yVal)


## 2 with Cross val
# clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
# scores = cross_val_score(clf, X, y, cv=4)
# print("Accuracy: (95 confidence) %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


## 3 with multiple metrics
# scoring = ['precision_macro', 'recall_macro']
# clf = svm.SVC(kernel="linear", C=1, random_state=0)
# scores = cross_validate(clf, X, y, scoring=scoring, cv=4)
# print(scores)

# scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

# # Optimum model as per grid search
# clf = svm.SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape='ovr', degree=1, gamma='auto', kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)
# scores = cross_validate(clf, X, y, scoring=scoring, cv=4)
# print(scores)

# prec = np.mean(scores['test_precision_macro'])
# recall = np.mean(scores['test_recall_macro'])
# f1 = np.mean(scores['test_f1_macro'])
# accuracy = np.mean(scores['test_accuracy'])

# print("Prec: ", prec)
# print("Recall: ", recall)
# print("f1: ", f1)
# print("accuracy: ", accuracy)




## Stratified k fold
# skf = StratifiedKFold(n_splits=10)
# for trainIndices, testIndices in skf.split(X, y):
#   xTrain = X[trainIndices]
#   yTrain = y[trainIndices]
#   xTest = X[testIndices]
#   yTest = y[testIndices]

#   # print("xTrain: ", xTrain.shape)
#   # print("yTrain: ", yTrain.shape)
#   # print("xTest: ", xTest.shape)
#   # print("yTest: ", yTest.shape)

#   scaler = preprocessing.StandardScaler().fit(xTrain)
#   xTrainTransformed = scaler.transform(xTrain)
#   clf = svm.SVC(C=1).fit(xTrainTransformed, yTrain)
#   xTestTransformed = scaler.transform(xTest)
#   scores = clf.score(xTestTransformed, yTest)
#   print(scores)

parameter_candidates_svm = [
  {'C': [1, 10, 100, 1000], 'degree': [1, 2, 3], 'class_weight': ['balanced', None]},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], 'class_weight': ['balanced', None]},
]

accuracies = []
f1Macros = []
maxAccuracy = 0
k = 10
for i in range(k):

  if k==1:
    xTrain, yTrain, xVal, yVal = splitAndOverSample(X, y)
  else:
    xTrain, yTrain, xVal, yVal = splitAndOverSample(X, y, k, i)

  # print("xTrain: ", xTrain.shape)
  # print("yTrain: ", yTrain.shape)
  # print("xVal: ", xVal.shape)
  # print("yVal: ", yVal.shape)

  print("Choosing optimum hyper parameters for training data")
  print("Internally performing 3 fold cross validation")

  # Choose optimum model on trainig data
  clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates_svm, n_jobs=-1, scoring='f1')
  clf.fit(xTrain, yTrain)

  # Extract the best model
  best = clf.best_estimator_
  print("Best Model found: ")

  # Predict the validation values
  yPred = best.predict(xVal)

  # PLot the values
  # plt.hist([yPred, yVal], align='left', rwidth=0.5, label=['Predicted', 'Actual'])
  # plt.legend()
  # plt.show()

  # Compute metrics
  accuracy = accuracy_score(yVal, yPred)
  f1Macro = f1_score(yVal, yPred, average="macro")

  if(accuracy > maxAccuracy):
    bestModel = best
    maxAccuracy = accuracy

  # evaluate the model
  accuracies.append(accuracy)
  f1Macros.append(f1Macro)

  print(i, ": Accuracy = ", accuracy)
  print(i, ": F1Macro = ", f1Macro)

print("Accuracies ")
print(accuracies)

print("F1s")
print(f1Macros)

print("Average accuracy: ", np.mean(accuracies))
print("Average F1 Macro: ", np.mean(f1Macros))

data = readData(filename='testData.csv')
X = createDesignMatrix(data)
xTest = featureNormalize(X)

y = createLabelVector(data)
yTest = np.squeeze(np.asarray(y))
yTest = np.array([1 if x>7 else 0 for x in y]);

yPred = bestModel.predict(xTest)

plt.hist([yPred, yTest], align='left', rwidth=0.5, label=['Predicted', 'Actual'])
plt.legend()
plt.show()

print("Accuracy on test set: ", accuracy_score(yTest, yPred))
print("F1Macro on test set: ", f1_score(yTest, yPred, average="macro"))