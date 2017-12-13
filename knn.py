from util import *
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from random import randint

#Trains the model and returns accuracy and a column matrix of the predicted labels
def trainModel(xtrain, ytrain, xtest, ytest, clf):
    clf.fit(xtrain, ytrain)
    yPrediction = clf.predict(xtest)
    accuracy = clf.score(xtest,ytest)
    return accuracy, yPrediction


#Take in the metric arrays and print the results
def printMetrics(averageAcc, averagePre):
    sum = 0
    for accuracy in averageAcc:        
        sum += accuracy 
    Acc = sum/len(averageAcc)

    print("Average accuracy is : ", Acc)
    print("_________________________")
    print("Precision is : ", averagePre)    
    

data = readData()


clf = neighbors.KNeighborsClassifier(n_neighbors=20)


print("Starting learning with ", len(data), " instances.")
x = createDesignMatrix(data)
y = createLabelVector(data)
if(TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD"):   
    print("Lenght of x : ", len(x))     
    acc = []
    pre = []
    numDataSplits = TRAINING_PARAMS['NUM_SPLITS']
    #print("Cross Validation used for splitting")
    for i in range(numDataSplits):
        xTrain, yTrain, xTest, yTest = splitUpDataCrossVal(x, y, numDataSplits, crossValIndex=i)
        averageAcc, yPrediction = trainModel(xTrain, yTrain, xTest, yTest, clf)
        print("Accuracy on fold", i, ":", averageAcc)
        acc.append(averageAcc)
else:
    #print("70/30 method used for splitting")
    xTrain, yTrain, xTest, yTest =splitData7030(x, y)
    averageAcc, yPrediction = trainModel(xTrain, yTrain, xTest, yTest)


printMetrics(acc, 0)

changeToTestData()
testData = readData()
print(data[1])
print(testData[1])
x = createDesignMatrix(testData)
y = createLabelVector(testData)

#yPrediction = clf.predict(x)
accuracy = clf.score(x,y)
print("\n\n___________________________")
print("Accuracy of test set:", accuracy)
print("___________________________")


