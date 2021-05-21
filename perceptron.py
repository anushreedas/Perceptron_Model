"""
perceptron.py

This program implements a perceptron model

@author: Anushree Das (ad1707)
"""
from utils import *

Perceptron = namedtuple('Perceptron',['eta', 'weightColMat', 'trace'])


def makePerceptron(eta,n,fn,trace):
    """
    Returns a Perceptron named tuple with the given parameters
    :param eta: learning rate
    :param n:   number of intended inputs of the perceptron
    :param fn:  initialization thunk
    :param trace: trace value
    :return:    Perceptron named tuple
    """
    return Perceptron(eta=eta, weightColMat=makeMatrix(n+1,1,fn), trace=trace)


def sigma(x):
    """
    Implements the step function for perceptron.
    It returns 1 if value is greater than 0 and
    returns 0 if value is less than or equal to 0
    :param x:   value
    :return:    0 or 1
    """
    if x > 0:
        return 1
    else:
        return 0


def applyPerceptron(perceptron,augColMatrix):
    """
    Returns the output of the perceptron - either a zero or a one.
    :param perceptron:  perceptron model
    :param augColMatrix:list of input vectors
    :return: 0 or 1
    """
    for element in augColMatrix.data:
        if element != 0 and element != 1:
            raise TypeError('The augmented column matrix should contain only 1s and 0s')

    # returns sigma(w.V) where w is weights of the model and V is a list of input vectors
    return sigma(dot(perceptron.weightColMat,augColMatrix))


def applyPerceptronVec(perceptron,inputVector):
    """
    Returns the output of the perceptron either a zero or a one.
    :param perceptron:  perceptron model
    :param augColMatrix:input vector
    :return:  0 or 1
    """
    for element in inputVector.data:
        if element != 0 and element != 1:
            raise TypeError('The input vector should contain only 1s and 0s')

    # convert input vector to column matrix and augment the column matrix
    augColMatrix = augmentColMat(colMatrixFromVector(inputVector))
    # returns sigma(w.v) where w is weights of the model and n is a input vector
    return sigma(dot(perceptron.weightColMat,augColMatrix))


def trainOnce(perceptron,inputVector,targetOutput):
    """
    Applies the perceptron learning rule to the perceptron model
    :param perceptron:  perceptron model
    :param inputVector: sample input vector
    :param targetOutput:target output
    :return: Boolean indicating whether or not the weights changed
    """
    if perceptron.trace == 2:
        print('On sample: input=',inputVector,'target=',targetOutput,',',perceptron)

    # convert input vector to column matrix and augment the column matrix
    augColMatrix = augmentColMat(colMatrixFromVector(inputVector))
    # predict output for input vector
    y0 = applyPerceptron(perceptron,augColMatrix)
    # get difference between the predicted output and the actual output
    delta = targetOutput - y0
    # find delta of weights
    deltaWeights = perceptron.eta * delta * augColMatrix
    # update weights
    setMat(perceptron.weightColMat,add(perceptron.weightColMat, deltaWeights))

    # return True if weights were changed else return False
    if sum(deltaWeights.data) != 0:
        return True
    else:
        return False


def andDataSetCreator():
    """
    Creates AND dataset for perceptron model
    :return: AND dataset
    """
    data = []
    for i in range(1,-1,-1):
        for j in range(1,-1,-1):
            data.append((Vector(data=[i,j]),i & j))
    return data


andDataSet = andDataSetCreator()


def orDataSetCreator():
    """
    Creates OR dataset for perceptron model
    :return: OR dataset
    """
    data = []
    for i in range(1,-1,-1):
        for j in range(1,-1,-1):
            data.append((Vector(data=[i,j]),i | j))
    return data


orDataSet = orDataSetCreator()


def trainEpoch(perceptron,dataset):
    """
    Trains the perceptron once for each entry in the data set
    :param perceptron:  perceptron model
    :param dataset:     data set on which to train on
    :return:            returns true, if any changes occur
    """
    updated = False
    # train for each row in dataset
    for sample in dataset:
        flag = trainOnce(perceptron, sample[0], sample[1])
        updated = updated or flag

    if perceptron.trace == 1:
        print('After epoch: ', perceptron)

    return updated


def train(perceptron,dataset,epochs):
    """
    Trains perceptron on given dataset iteratively and
    terminates when either there is no change or
    the number of epochs exceeds the bound.
    :param perceptron:  perceptron model
    :param dataset:     data set on which to train on
    :param epochs:      number of training epochs
    :return:            None
    """
    iteration = 0
    updated = True

    # terminates when either there is no change or the number of epochs exceeds the bound
    while iteration <= epochs and updated:
        updated = trainEpoch(perceptron,dataset)
        iteration += 1
