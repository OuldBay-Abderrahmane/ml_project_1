import numpy as np

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
