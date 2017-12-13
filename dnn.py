# from keras.models import Sequential
# from keras.layers import Dense
import keras
from keras import datasets
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.regularizers import l2

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
        
        clear_output(wait=True)
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

xTrain, yTrain, xTest, yTest = splitData7030(X, y)
print(xTrain.shape)


# model = Sequential()
# model.add(Dense(units=64, activation='relu', input_dim=100))
# model.add(Dense(units=10, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])

# model.fit(xTrain, yTrain, epochs=5, batch_size=32)


inputs = keras.layers.Input(shape=xTrain.shape)
x = Flatten()(inputs)
x = Dense(100, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(400, activation='relu')(x)
x = Dense(400, activation='relu')(x)
x = Dense(400, activation='relu')(x)


# 10 categories
predictions = Dense(10, activation='softmax')(x)

# we create the model 
model = keras.models.Model(inputs=inputs, outputs=predictions)

# SGD - Stoachaistc Gradient Model
# lr: float >= 0. Learning rate.
# momentum: float >= 0. Parameter updates momentum.
# decay: float >= 0. Learning rate decay over each update.
# nesterov: boolean. Whether to apply Nesterov momentum.
opt = keras.optimizers.SGD(lr=0.01, decay=1.1e-6, momentum=0.9, nesterov=True)

# setup the optimisation strategy
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()


pltCallBack = PlotLossAccuracy()

# and train
model.fit(xTrain, yTrain,
          batch_size=256, epochs=60, validation_data=(xTest, yTest)) #, 
          # callbacks=[pltCallBack])