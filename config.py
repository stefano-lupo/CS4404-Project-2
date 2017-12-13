TRAINING_PARAMS = dict(
    NORMALIZE_METHOD = "MINMAX",            # How features should be normalized
    DESIRED_NUM_INSTANCES = 100000,         # Specify max number of instances (None uses all instances)
    SPLIT_METHOD = "KFOLD",                 # One of "70/30" or "KFOLD"
    NUM_SPLITS = 10,                        # K in K fold cross validation
    LEARNING_RATE = 0.1,                    # Stepsize for gradient descent
    TRAINING_EPOCHS = 100,                  # Number of iterations of gradient descent training
    BALANCE_SAMPLING = 'OVER',
    IS_KNN_LABEL_STRING = False,            # If predicted string categorical data, set to True
    KNN_CLASS_THRESHOLD = None,             # The accepted deviation from true y value for numeric classification                                # Can be None for exact classification
    K = 2                                   # Number of nearest neighbours to use
)


# Specify a list of features using the FEATURES field,
# If these features are to be ommited then set OMIT_FEATURES to True
# If they are to be the features used for the inputs, set OMIT_FEATURES to False

# Available Features
# "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
# "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"

# White Wine Dataset
WHITE_WINE = dict(
    FILE_NAME = "winequality-white.csv",
    DELIMETER = ";",
    FEATURES = ["quality"],
    OMIT_FEATURES = True,
    LABEL = "quality"
)


###########################################
####                 KNN            #######
###########################################
WHITE_WINE_KNN = dict(WHITE_WINE)


# Select the dataset to be used by the algorithm
# Be sure to sure to use the <>_KNN datasets when using K nearest neighbours 
# then simply run `python <algorithm-script.py>` to get the results.
ACTIVE_DATASET = WHITE_WINE


