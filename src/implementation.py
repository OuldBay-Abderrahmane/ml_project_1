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


def gradient_descent(y, tx, initial_w, max_iters, gamma):
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
