import numpy as np

def featureSelection(data, targetSize=50):
    """Takes in an array of samples and trims off features
    with low appearance rates until targetSize or less remain

    Arguments:
        data: the NumPy array of samples, each sample being an array of integers
        targetSize: the target number of features, default 50
    
    Returns:
        NumPy array of samples reduced to only the targetSize most frequent features
    """
    numFeatures = len(data[0])
    numToRemove = numFeatures - targetSize

    # If nothing to remove, we're done
    if numToRemove <= 0:
        return data
    
    data = data.T

    featureCounts = []
    for feature in data:
        featureCounts.append(sum([0 if x == 0 else 1 for x in feature]))

    indexesToDelete = []
    while numToRemove > 0:
        min = 0
        for i, _ in enumerate(featureCounts):
            if featureCounts[i] < featureCounts[min]:
                min = i
        indexesToDelete.append(min)
        featureCounts[min] = len(data[0])+1
        numToRemove -= 1

    data = [_ for i, _ in enumerate(data) if i not in indexesToDelete]

    return np.array(data).T


def preprocess(X, Y, C0, C1):
    """Takes in a dataset from keras.datasets.mnist in two arrays, 
    one with samples and the other with the labels at corresponding indices, 
    and applies preprocessing rules, including reducing the samples to only those
    labelled with C0 or C1, flattening the 2D samples into 1D arrays, and normalizing
    sample values into the [0, 1] range.

    Arguments:
        X: Python list of MNIST samples
        Y: Python list of MNIST labels
        C0: The label of class 0
        C1: The label of class 1
    
    Returns:
        X: The preprocessed sample set as a NumPy array
        Y: The preprocessed label set as a NumPy array
    """
    # Filter the datasets down to just the required classes
    X = [_ for i, _ in enumerate(X) if Y[i] == C0 or Y[i] == C1]
    Y = [y for y in Y if y == C0 or y == C1]
    
    # Flatten the 2D representations of the samlpes into 1D arrays
    X = np.reshape(X, (len(X), len(X[0])*len(X[0])))

    # Normalize sample values to be between 0 and 1
    # X = [[x/256 for x in sample] for sample in X]
    
    # Normalize class labels to be 0 and 1
    Y = np.fromiter((0 if y == C0 else 1 for y in Y), int)

    return np.array(X), Y
