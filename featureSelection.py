from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2

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
print("X: ", X.shape)
print("y: ", y.shape)

varX = np.var(X, axis=0)
print(varX)


sel = SelectKBest(chi2, k=4).fit(X, y)
indices = sel.get_support(indices=True)
print(indices)

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





