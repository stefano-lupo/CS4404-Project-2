from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score, recall_score
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

data = readData('testData.csv')
X = createDesignMatrix(data)
X = featureNormalize(X)

y = createLabelVector(data)
y = np.squeeze(np.asarray(y))
print("X: ", X.shape)
print("y: ", y.shape)


parameter_candidates_svm = [
  {'C': [1, 10, 100, 1000], 'degree': [1, 2, 3], 'class_weight': ['balanced', None]},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], 'class_weight': ['balanced', None]},
]

parameter_candidates_log_reg = [
  {'C': [1,  100, 1000], 'fit_intercept': [True, False], 'class_weight': ['balanced', None],
  'multi_class': ['ovr', 'multinomial'], 'penalty': ['l2'], 'intercept_scaling': [1,2], 'solver': ['newton-cg']}
]


clf = GridSearchCV(estimator=linear_model.LogisticRegression(), param_grid=parameter_candidates_log_reg, n_jobs=-1, scoring='f1_macro')
clf.fit(X, y)   
print("Logistic Regression")
print(clf.best_estimator_)
print(clf.best_estimator_.predict(X))
"""
Gives (when trained on whole dataset) maximised for f1-macro
LogisticRegression(C=1000, class_weight='balanced', dual=False,
          fit_intercept=False, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='newton-cg', tol=0.0001, verbose=0, warm_start=False)
"""


# clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates_svm, n_jobs=-1, scoring='f1_macro')
# clf.fit(X, y)
# print("SVC")
# print(clf.best_estimator_)
"""
Gives (when all data used - obviously shouldnt be the case) - optimised for f1-macro
SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=1, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

