import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import utils
import multi_class as mc


def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """

    m = X.shape[0]
    X = np.hstack([np.ones((m, 1)), X])
    a = mc.sigmoid(np.dot(X, theta1.T))
    a = np.hstack([np.ones((m, 1)), a])
    a2 = mc.sigmoid(np.dot(a, theta2.T))
    J = -1/m * np.sum(y[:, np.newaxis] * np.log(a2) + (1 - y)[:, np.newaxis] * np.log(1 - a2))

    return J


def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """

    m = X.shape[0]
    X = np.hstack([np.ones((m, 1)), X])
    a = mc.sigmoid(np.dot(X, theta1.T))
    a = np.hstack([np.ones((m, 1)), a])
    a2 = mc.sigmoid(np.dot(a, theta2.T))
    
    delta = a2 - y

    grad1 = 1/m * np.dot((np.dot(delta, theta2) * a * (1 - a))[:, 1:].T, X)
    grad1[:, 1:] += lambda_/m * theta1[:, 1:]

    grad2 = 1/m * np.dot(delta.T, a)
    grad2[:, 1:] += lambda_/m * theta2[:, 1:]

    J = -1/m * np.sum(y * np.log(a2) + (1 - y) * np.log(1 - a2)) + lambda_/(2*m) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    
    return (J, grad1, grad2)


def main():
    utils.checkNNGradients(backprop, 0.1)


def testCost():
    data = sio.loadmat('data/ex3data1.mat', squeeze_me=True)
    X = data['X']
    y = data['y']

    weights = sio.loadmat('data/ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']

    cost_ = cost(theta1, theta2, X, y, 0)
    print('Cost at parameters (loaded from ex3weights):', cost_)
    print('Expected cost: 0.287629')


if __name__ == "__main__":
    main()
    testCost()