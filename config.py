TRAINING_PARAMS = dict(
    NORMALIZE_METHOD = "MINMAX",            # How features should be normalized
    DESIRED_NUM_INSTANCES = 100000,         # Specify max number of instances (None uses all instances)
    SPLIT_METHOD = "70/30",                 # One of "70/30" or "KFOLD"
    NUM_SPLITS = 10,                        # K in K fold cross validation
    OVER_SAMPLE = True,                     # Whether or not to oversample
    USE_OS_DICT = True                     # Whether or not to use the oversample amount dict or even
)

SHOW_PLOTS = False

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


# White Wine Dataset with feature engineering
WHITE_WINE_OPT = dict(
    FILE_NAME = "winequality-white.csv",
    DELIMETER = ";",
    FEATURES = ["volatile acidity", "chlorides", "density", "alcohol"],
    OMIT_FEATURES = False,
    LABEL = "quality"
)

# Wine dataset without test data removed
WHITE_WINE_FULL = dict(
    FILE_NAME = "winequality-white-full.csv",
    DELIMETER = ";",
    FEATURES = ["quality"],
    OMIT_FEATURES = True,
    LABEL = "quality"
)


# Select the dataset to be used by the algorithm
# Be sure to sure to use the <>_KNN datasets when using K nearest neighbours 
# then simply run `python <algorithm-script.py>` to get the results.
ACTIVE_DATASET = WHITE_WINE


