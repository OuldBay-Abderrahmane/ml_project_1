import numpy as np
from helpers import batch_iter

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.sum((y - tx @ w)**2) / (2 * tx.shape[0])


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape (D,), D is the number of features.
        mse: scalar.
    """
    wTX = np.dot(tx.T, tx)
    wTY = np.dot(tx.T, y)
    w = np.linalg.solve(wTX, wTY)
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression with a given lambda (regularization strength).

    Args:
        y: Target variable (numpy array).
        tx: Input features with polynomial basis (numpy array).
        lambda_: Regularization strength (scalar).

    Returns:
        w_ridge: Ridge regression weights (numpy array).
    """
    lambda_prime = 2 * len(y) * lambda_
    A = tx.T.dot(tx) + lambda_prime * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w_ridge = np.linalg.solve(A, b)

    return w_ridge


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    # MSE

    error = y - tx @ w
    return - 1/tx.shape[0] * error @ tx


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_loss(y, tx, w)
        w_g = compute_gradient(y, tx, w)

        # update w by gradient
        w = w - gamma * w_g

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just a few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    return -tx.T.dot(y - tx.dot(w))


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the step size

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            stochastic_gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w -= gamma * stochastic_gradient
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            losses.append(loss)

        ws.append(w)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws

def sigmoid(t):
    """apply sigmoid function on t.

    """
    return 1/(1 + np.exp(-t)) 


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    z = np.dot(tx, w)
    return (1/y.shape[0])*np.sum(np.log(1 + np.exp(z)) - y * z)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.
    """
    z = tx.dot(w) 
    sigmoids = sigmoid(z)

    return 1/y.shape[0] * tx.T.dot(sigmoids - y)


def logistic_regression(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss.
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    return loss, gradient

#REGULARIZED LOGISTIC REGRESSION
def compute_loss_reg(y, tx, w, lambda_):
    """compute the cost"""
    l2_regularization = (lambda_ / 2) * np.sum(w**2)

    return (- calculate_loss(y, tx, w) + (1/y.shape[0])*l2_regularization)

def compute_gradient_reg(y, tx, w, lambda_):
    """compute the gradient of loss"""
    return calculate_gradient(y, tx, w) + lambda_ * w
   
def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    """return the loss, gradient of the loss, and hessian of the loss.
    """
    loss = compute_loss_reg(y, tx, w, lambda_)
    gradient = calculate_gradient(y, tx, w, lambda_)
    return loss, gradient