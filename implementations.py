import numpy as np
from helpers import batch_iter

W = "np.array([0.5, 1.0])"
Y = "np.array([0.1, 0.3, 0.5])"
TX = "np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])"
MAX_ITERS = "2"
GAMMA ="0.1"
def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.mean((y - tx.dot(w)) ** 2)


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape (D,), D is the number of features.
        mse: scalar.
    >>> least_squares({Y}, {TX})
    (0.026942, np.array([0.218786, -0.053837]))
    """
    wTX = np.dot(tx.T, tx)
    wTY = np.dot(tx.T, y)
    w = np.linalg.lstsq(wTX, wTY)
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    f"""Ridge regression with a given lambda (regularization strength).

    Args:
        y: Target variable (numpy array).
        tx: Input features with polynomial basis (numpy array).
        lambda_: Regularization strength (scalar).

    Returns:
        w_ridge: Ridge regression weights (numpy array).

    >>> ridge_regression({Y}, {TX}, 0)
    (0.026942, np.array([0.218786, -0.053837]))
    >>> ridge_regression({Y}, {TX}, 1)
    (0.03175, np.array([0.054303, 0.042713]))
    """
    lambda_prime = 2 * len(y) * lambda_
    A = tx.T.dot(tx) + lambda_prime * np.identity(tx.shape[0])
    b = tx.T.dot(y)
    w_ridge = np.linalg.lstsq(A, b)

    return w_ridge, compute_loss(y, tx,w_ridge)


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
    f"""The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    >>> mean_squared_error_gd({Y}, {TX}, {W}, 0, {GAMMA})
    (0.026942, np.array([0.218786, -0.053837]))
    >>> mean_squared_error_gd({Y}, {TX}, {W}, {MAX_ITERS}, {GAMMA})
    (0.051534, np.array([-0.050586, 0.203718]))
    
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
    return ws, losses


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
    f"""The Stochastic Gradient Descent algorithm (SGD).

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
    >>> test_mean_squared_error_sgd({Y[:1]}, {TX[:1]}, {W}, 0, {GAMMA})
    (0.844595, np.array([0.063058, 0.39208]))
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
    return ws, losses

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


def logistic_regression(y, tx, w, max_iter, gamma):
    f"""return the loss, gradient of the loss, of the loss.
    >>> logistic_regression({Y}, {TX}, {W}, 0, {GAMMA})
    (1.533694, np.array([0.463156, 0.939874]))
    >>> logistic_regression({Y}, {TX}, {W}, {MAX_ITERS}, {GAMMA})
    (1.348358, np.array([0.378561, 0.801131]))
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    return gradient, loss

#REGULARIZED LOGISTIC REGRESSION
def compute_loss_reg(y, tx, w, lambda_):
    """compute the cost"""
    l2_regularization = (lambda_ / 2) * np.sum(w**2)

    return (- calculate_loss(y, tx, w) + (1/y.shape[0])*l2_regularization)

def compute_gradient_reg(y, tx, w, lambda_):
    """compute the gradient of loss"""
    return calculate_gradient(y, tx, w) + lambda_ * w
   
def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    f"""return the loss, gradient of the loss, and hessian of the loss.
    >>> logistic_regression({Y}, {TX}, {W}, 0, {GAMMA})
    (1.407327, np.array([0.409111, 0.843996]))
    >>> logistic_regression({Y}, {TX}, {W}, {MAX_ITERS}, {GAMMA})
    (0.972165, np.array([0.216062, 0.467747]))
    """
    loss = compute_loss_reg(y, tx, w, lambda_)
    gradient = compute_gradient_reg(y, tx, w, lambda_)
    return gradient, loss

