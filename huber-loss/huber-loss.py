import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss.

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    delta : float
        Threshold parameter

    Returns
    -------
    float
        Mean Huber loss
    """

    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Compute prediction error
    e = y_true - y_pred
    abs_e = np.abs(e)

    # Apply piecewise Huber formula
    quadratic = 0.5 * e**2
    linear = delta * (abs_e - 0.5 * delta)

    loss = np.where(abs_e <= delta, quadratic, linear)

    # Return mean loss
    return float(np.mean(loss))


# =========================
# Example 1
# =========================
y_true1 = [1, 2, 3]
y_pred1 = [1.5, 1.7, 2.5]

print("Example 1 Loss:", huber_loss(y_true1, y_pred1, delta=1.0))


# =========================
# Example 2
# =========================
y_true2 = [0, 5]
y_pred2 = [2, 8]

print("Example 2 Loss:", huber_loss(y_true2, y_pred2, delta=1.0))


# =========================
# Example 3
# =========================
y_true3 = [1, 2]
y_pred3 = [1, 2]

print("Example 3 Loss:", huber_loss(y_true3, y_pred3))