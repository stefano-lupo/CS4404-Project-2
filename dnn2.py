from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import datasets
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.regularizers import l2

from collections import Counter
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

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




# Define some useful functions
class PlotLossAccuracy(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(int(self.i))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
        self.i += 1
        
    def show_plots(self):
        plt.figure(figsize=(16, 6))
        plt.plot([1, 2])
        plt.subplot(121) 
        plt.plot(self.x, self.losses, label="train loss")
        plt.plot(self.x, self.val_losses, label="validation loss")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Model Loss')
        plt.legend()
        plt.subplot(122)         
        plt.plot(self.x, self.acc, label="training accuracy")
        plt.plot(self.x, self.val_acc, label="validation accuracy")
        plt.legend()
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title('Model Accuracy')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show();








data = readData()
X = createDesignMatrix(data)
X = featureNormalize(X)

y = createLabelVector(data)
y = np.squeeze(np.asarray(y))
print(X.shape)
print(y.shape)

xTrain, yTrain, xTest, yTest = splitData7030(X, y)
print(xTrain.shape)
print(yTrain.shape)

from keras.utils import to_categorical
yTrain = to_categorical(yTrain, num_classes=10)
yTest = to_categorical(yTest, num_classes=10)
print("After binary encoding y: ")
print(yTrain.shape)
print(yTest.shape)


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=xTrain.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='softmax'))
model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

pltCallBack = PlotLossAccuracy()

model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=200, batch_size=10, callbacks=[pltCallBack])

pltCallBack.show_plots()

predictions = model.predict(xTest)
print("Predicter: ")
print(predictions[0], "\n")

print("Actual: ")
print(yTest, "\n")

# evaluate the model
scores = model.evaluate(xTest, yTest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
