
# coding: utf-8

# In[1]:


from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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
    TRAINING_PARAMS,
    splitAndOverSample
)


NUM_CLASSES = 10
showPlots = False


data = readData()
X = createDesignMatrix(data)
# X = featureNormalize(X)

y = createLabelVector(data)
y = np.squeeze(np.asarray(y))
# y = np.array([1 if x>7 else 0 for x in y]);



parameter_candidates_log_reg = [
  {'C': [1, 10, 100, 500]}
]


learning = True
learnedVsDefault = {'learned': dict(), 'default': dict()}






# In[2]:


for i in range(2):

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
      print("Learning Model")
      clf = GridSearchCV(estimator=linear_model.LogisticRegression(random_state=0), param_grid=parameter_candidates_log_reg, n_jobs=-1, scoring="f1_macro")
      clf.fit(xTrain, yTrain)
      clf = clf.best_estimator_
      print("Optimum C: ", clf.C)
    else:
      print("Using default LR Model")
      clf = linear_model.LogisticRegression(random_state=0)
      clf.fit(xTrain, yTrain)

    # Predict the validation values
    yPred = clf.predict(xVal)

    # PLot the values
#     plt.hist([yPred, yVal], align='left', rwidth=0.5, label=['Predicted', 'Actual'])
#     plt.legend()
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



# In[3]:


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

    print("\nMacro")
    for key, value in macro.items():
        print(key, " ", np.mean(value))

    print("Micro")
    for key, value in micro.items():
        print(key, " ", np.mean(value))

    print("\n")


# In[66]:


learned = learnedVsDefault['learned']
default = learnedVsDefault['default']



learnedMetrics = [np.mean(learned['accuracies']), np.mean(learned['kappas'])]
for (key, val) in learned['macro'].items():
    learnedMetrics.append(np.mean((val)))

print(learnedMetrics)

defaultMetrics = [np.mean(default['accuracies']), np.mean(default['kappas'])]
for (key, val) in default['macro'].items():
    defaultMetrics.append(np.mean(val))
    
print(defaultMetrics)


plt.xticks([1,2,3,4,5], ['Accuracy', 'Kappa', 'F1', 'Precision', 'Recall'])
plt.xlabel('Metric')
plt.ylabel('Value')
plt.bar([1, 2, 3, 4, 5], learnedMetrics, label="Learned", alpha=1)
plt.bar([1, 2, 3, 4, 5], defaultMetrics, label="Default", alpha=1)
plt.legend()
plt.show()




