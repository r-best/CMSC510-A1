import numpy as np

# Sample size of training set can be provided as a percent between 0 and 1,
#   otherwise the full training set will be used
def parseArgs(argv):
    """Processes command line arguments, currently the only one is the sample
    size to use for training, given as a percentage value in the range (0, 1]

    Arguments:
        argv: array-like
            The arguments obtained from sys.argv
    
    Returns:
        sampleSize: float
            The percentage of training samples to be used
    """
    sampleSize = 1
    if len(argv) > 1:
        try:
            temp = float(argv[1])

            if temp <= 0 or temp > 1:
                raise ValueError
            else:
                sampleSize = temp
        except ValueError:
            print("WARN: Invalid sample size, must be a decimal value in range (0, 1]. Using full sample set for this run.")

    return sampleSize


def featureSelection(train, test, targetSize=50):
    """Takes in an array of training data and an array of testing data,
    reduces their feature size down to targetSize by removing the features
    that occur the least often in the training set

    Arguments:
        train: array-like (2D)
            Array of training samples, each sample being an array of features
        test: array-like (2D)
            Array of test samples, same format as train
        targetSize: int
            Target number of features, default 50
    
    Returns:
        Train and test reduced to targetSize features
    """
    numFeatures = len(train[0])
    numToRemove = numFeatures - targetSize

    # If nothing to remove, we're done
    if numToRemove <= 0:
        return train, test
    
    train = np.transpose(train)

    featureCounts = []
    for feature in train:
        featureCounts.append(sum([0 if x == 0 else 1 for x in feature]))

    indexesToDelete = []
    while numToRemove > 0:
        min = 0
        for i, _ in enumerate(featureCounts):
            if featureCounts[i] < featureCounts[min]:
                min = i
        indexesToDelete.append(min)
        featureCounts[min] = len(train[0])+1
        numToRemove -= 1

    train = [_ for i, _ in enumerate(train) if i not in indexesToDelete]
    test = [[_ for i, _ in enumerate(sample) if i not in indexesToDelete] for sample in test]

    return np.array(train).T, np.array(test)


def preprocess(X, Y, C0, C1):
    """Takes in a dataset from keras.datasets.mnist in two arrays, 
    one with samples and the other with the labels at corresponding indices, 
    and applies preprocessing rules, including reducing the samples to only those
    labelled with C0 or C1, flattening the 2D samples into 1D arrays, and normalizing
    sample values into the [0, 1] range.

    Arguments:
        X: array-like (2D)
            Array of MNIST samples
        Y: array-like (1D)
            Array of MNIST labels
        C0: int
            The label of class 0
        C1: int
            The label of class 1
    
    Returns:
        X: ndarray
            The preprocessed sample set as a NumPy array
        Y: ndarray
            The preprocessed label set as a NumPy array
    """
    # Filter the datasets down to just the required classes
    X = [_ for i, _ in enumerate(X) if Y[i] == C0 or Y[i] == C1]
    Y = [y for y in Y if y == C0 or y == C1]
    
    # Flatten the 2D representations of the samples into 1D arrays
    X = np.reshape(X, (len(X), len(X[0])**2))

    # Normalize sample values to be between 0 and 1
    # X = [[x/256 for x in sample] for sample in X]
    
    # Normalize class labels to be 0 and 1
    Y = np.fromiter((0 if y == C0 else 1 for y in Y), int)

    return np.array(X), Y
