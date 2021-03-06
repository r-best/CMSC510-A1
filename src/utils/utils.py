"""
Robert Best
"""

import numpy as np
from sklearn import metrics


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
    # X = [[x/256 for x in sample] for sample in X] # REMOVED, was tanking accuracy
    
    # Normalize class labels to be 0 and 1
    Y = np.fromiter((0 if y == C0 else 1 for y in Y), int)

    return np.array(X), Y


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

    # If nothing to remove, we're done
    if numFeatures <= targetSize:
        return train, test
    
    train = np.transpose(train)

    # Sum number of times each feature occurs across training set
    featureCounts = []
    for feature in train:
        featureCounts.append(sum([0 if x == 0 else 1 for x in feature]))

    # Obtain the targetSize (50) most occurring features
    maxIndexes = []
    while targetSize > 0:
        max = 0
        for i, _ in enumerate(featureCounts):
            if featureCounts[i] > featureCounts[max]:
                max = i
        maxIndexes.append(max)
        featureCounts[max] = -1
        targetSize -= 1

    # Filter train and test sets down to just the target features
    train = [_ for i, _ in enumerate(train) if i in maxIndexes]
    test = [[_ for i, _ in enumerate(sample) if i in maxIndexes] for sample in test]

    return np.array(train).T, np.array(test)


def evaluate(labels, gold):
    """Takes in an array of predicted labels and the corresponding
    gold standard and calculates precision, recall, and accuracy.

    Arguments:
        labels: array-like (1D)
            The predicted labels, either 1 or 0
        gold: array-like (1D)
            The correct labels
    
    Returns:
        None
    """
    labels = list(labels)

    # Get confusion matrix, thanks scikit-learn!
    conf_matrix = metrics.confusion_matrix(gold, labels)
    correct = conf_matrix[0][0]+conf_matrix[1][1] # Total correct is true 0 + true 1
    conf_matrix = [[str(x)+"  " if x <= 9 else str(x)+" " if x <= 99 else str(x) for x in row] for row in conf_matrix]
    print("---------------------------------------")
    print("|                        Actual       |")
    print("|          ---------------------------|")
    print("|          |     |    0    |     1    |")
    print("|          |-----+---------+----------|")
    print("|          |  0  |   {}   |    {}   |".format(conf_matrix[0][0], conf_matrix[0][1]))
    print("|Predicted |     |         |          |")
    print("|          |  1  |   {}   |    {}   |".format(conf_matrix[1][0], conf_matrix[1][1]))
    print("---------------------------------------")
    
    # Get precision/recall/f-measure for both classes with one method call, thanks scikit-learn!
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(gold, labels)
    print("Class 0 Precision: {:.3f}".format(precision[0]))
    print("Class 0 Recall: {:.3f}".format(recall[0]))
    print("Class 0 F-Measure: {:.3f}".format(fscore[0]))
    print("Class 1 Precision: {:.3f}".format(precision[1]))
    print("Class 1 Recall: {:.3f}".format(recall[1]))
    print("Class 1 F-Measure: {:.3f}".format(fscore[1]))
    print("Accuracy: {}/{} = {:.3f}%".format(correct, len(gold), correct/len(gold)*100))
