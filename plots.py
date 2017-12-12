import numpy as np
import matplotlib.pyplot as plt

from util import (
    readData,
    createDesignMatrix,
    createLabelVector,
    printM,
    splitData7030,
    splitUpDataCrossVal,
    featureNormalize,
    multiclassToBinaryClass,
    ACTIVE_DATASET
)

from config import TRAINING_PARAMS

data = readData()
points = createDesignMatrix(data)
classes = createLabelVector(data)
numFeatures = points.shape[1]

print(np.arange(np.min(classes), np.max(classes)+1))

# Plot the distribution of classes
plt.hist(classes, bins=np.arange(np.min(classes), np.max(classes)+1))
plt.show()

# Plot the rating vs each of the input features
# for i in range(numFeatures):
#     print(i)
#     plt.plot(points[:, i], classes, 'ro')
#     plt.xlabel("Feature %d" % i)
#     plt.ylabel(ACTIVE_DATASET['LABEL'])
#     plt.show()








