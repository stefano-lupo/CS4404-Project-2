from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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
      ratioDict = {3: max(400, counts[3]), 4: max(700, counts[4]), 5: counts[5],
                    6: counts[6], 7: counts[7], 8: max(700, counts[8]), 9: max(400, counts[9])}

      # Oversample the training data
      # xTrainOS, yTrainOS = RandomOverSampler(random_state=0, ratio=ratioDict).fit_sample(xTrain, yTrain)
      xTrainOS, yTrainOS = RandomOverSampler(random_state=0).fit_sample(xTrain, yTrain)
      # xTrainOS, yTrainOS = SMOTE(k_neighbors=1, ratio=ratioDict).fit_sample(xTrain, yTrain)
      # xTrainOS, yTrainOS = ADASYN(n_neighbors=2, ratio=ratioDict).fit_sample(xTrain, yTrain)

      print('Oversampling')
      print('Rating distribution: ', sorted(Counter(yTrainOS).items()))
      

      if showPlots:
        # Show distribution of classes after over sampling
        plt.hist([yTrain, yTrainOS], bins=range(3, 11), align='left', rwidth=0.5, label=['No Oversampling', 'Oversampling'])
        plt.legend()
        plt.show()

      return xTrainOS, yTrainOS, xVal, yVal
    else:
      return xTrain, yTrain, xVal, yVal

data = readData()
X = createDesignMatrix(data)
# X = featureNormalize(X)

y = createLabelVector(data)
y = np.squeeze(np.asarray(y))

# y = np.array([1 if x>7 else 0 for x in y]);
print("X: ", X.shape)
print("y: ", y.shape)


parameter_candidates_rf = [
  {'n_estimators': [20, 40, 60, 80, 100 ] }
]

learning = False
learnedVsDefault = {'learned': dict(), 'default': dict()}


for i in range(2):
  print("\n\nLearning = ", learning)
  accuracies = []
  kappas = []
  micro = {'f1': [], 'precision': [], 'recall': []}
  macro = {'f1': [], 'precision': [], 'recall': []}
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

    # Choose optimum model on trainig data

    if(learning):
      clf = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=parameter_candidates_rf, n_jobs=-1, scoring="f1_macro")
      clf.fit(xTrain, yTrain)

      print("Optimum number of estimators: ", clf.best_estimator_.n_estimators)
      clf = clf.best_estimator_
    else:
      print("Using default RF")
      clf = RandomForestClassifier()
      clf.fit(xTrain, yTrain)

    # Predict the validation values
    yPred = clf.predict(xVal)

    # PLot the values
    plt.hist([yPred, yVal], align='left', rwidth=0.5, label=['Predicted', 'Actual'])
    plt.legend()
    # plt.show()

    # Compute metrics
    accuracy = accuracy_score(yVal, yPred)
    accuracies.append(accuracy)

    kappa = cohen_kappa_score(yVal, yPred)
    kappas.append(kappa)

    # Compute macro metrics
    macro['f1'].append(f1_score(yVal, yPred, average="macro"))
    macro['recall'].append(recall_score(yVal, yPred, average="macro"))
    macro['precision'].append(precision_score(yVal, yPred, average="macro"))

    # Compute micro metrics
    micro['f1'].append(f1_score(yVal, yPred, average="micro"))
    micro['recall'].append(recall_score(yVal, yPred, average="micro"))
    micro['precision'].append(precision_score(yVal, yPred, average="micro"))

  if(learning):
    learnedVsDefault['learned'] = {'micro': micro, 'macro': macro, 'accuracies': accuracies, 'kappas': kappas}
    learning = False
  else:
    learnedVsDefault['default'] = {'micro': micro, 'macro': macro, 'accuracies': accuracies, 'kappas': kappas}
    learning = True

for x in range(2):
  if(learning):
    key = 'learned'
    learning = False
  else:
    key = 'default'
    learning = True

  accuracies = learnedVsDefault[key]['accuracies']
  kappas = learnedVsDefault[key]['kappas']
  macro = learnedVsDefault[key]['macro']
  micro = learnedVsDefault[key]['micro']

  print("\n\nLearning = ", not learning)
  print("Average kappa: ", np.mean(kappas))
  print("Average accuracy: ", np.mean(accuracies))
  print("Macro")
  for key, value in macro.items():
    print(key, " ", np.mean(value))

  print("\n")
  print("Micro")
  for key, value in micro.items():
    print(key, " ", np.mean(value))

  print("\n")
