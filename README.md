# Perceptron_Model

Implements a perceptron model.

Usage:
## Create Perceptron:
perceptron = makePerceptron(eta, n, fn, trace)
where eta is learning rate, n is number of intended inputs of the perceptron, fn is initialization thunk and trace is trace value.

## Train Perceptron:
train(perceptron,dataset,epochs)

## Apply on single sample input:
applyPerceptronVec(perceptron,inputVector)

