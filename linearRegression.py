####
####Loading in Data and organising it
####
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

#import util functions
from util import *

# Prepends a column of ones to design matrix to represent bias term 
def prependBiasTerm(dataset):
  numInstances = dataset.shape[0]
  numFeatures = dataset.shape[1]
  X = np.reshape(np.c_[np.ones(numInstances),dataset],[numInstances,numFeatures + 1])
  return X

###
###Using sklearn to train
###
def trainModel(xTrain, yTrain):
  # Create linear regression object
  regr = linear_model.LinearRegression()
  # Train the model using the training sets
  regr.fit(xTrain, yTrain)
  yPrediction = regr.predict(xTest)
  return yPrediction

def evaluateModel(yTest, yPrediction):
  
  # The mean squared error
  mse = mean_squared_error(yTest, yPrediction)
  # The mean absolute error
  mae = mean_absolute_error(yTest, yPrediction)
  return mse, mae

def plotReg(yTest, yPrediction):
  # Plot outputs
  plt.plot(xTest[:,1], yTest, "ro" )
  plt.plot(xTest[:,1], yPrediction, "bo")
  # plt.plot(yTest, yPrediction, "ro")

  plt.xlabel("Actual Value")
  plt.ylabel("Predicted Value")
  plt.xticks(())
  plt.yticks(())

  plt.show()


#Use the util functions to read in the dataset and create required matrices
data = readData()
x = createDesignMatrix(data)
y = createLabelVector(data)


#Normalize the features
x = featureNormalize(x)
y = featureNormalize(y)

# Prepends a column of ones to design matrix to represent bias term 
x = prependBiasTerm(x)



if(TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD"):        
    mse = []
    mae = []
    numDataSplits = TRAINING_PARAMS['NUM_SPLITS']
    print("Cross Validation used for splitting")
    for i in range(numDataSplits):
        xTrain, yTrain, xTest, yTest = splitUpDataCrossVal(x, y, numDataSplits, crossValIndex=i)
        predictedY = trainModel(xTrain, yTrain)
        currentMSE, currentMAE = evaluateModel(yTest, predictedY)
        mse.append(currentMSE)
        mae.append(currentMAE)
    averageMSE = np.mean(mse)
    averageMAE = np.mean(mae)
else:
    print("70/30 method used for splitting")
    xTrain, yTrain, xTest, yTest =splitData7030(x, y)
    predictedY = trainModel(xTrain, yTrain)
    averageMSE, averageMAE = evaluateModel(yTest, predictedY)

# plotReg(xTrain, predictedY)

print("Average mean squared error: ", averageMSE)
print("Average mean absolute error: ", averageMAE)