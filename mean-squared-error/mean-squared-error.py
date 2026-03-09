import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Compute Mean Squared Error (MSE)

    Parameters
    ----------
    y_pred : array-like
        Predicted values (N,)

    y_true : array-like
        True target values (N,)

    Returns
    -------
    float
        Mean Squared Error
        Returns None if shapes do not match
    """

    # Convert inputs to NumPy arrays
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    # Check if shapes match
    if y_pred.shape != y_true.shape:
        return None

    # Compute squared error
    squared_error = (y_pred - y_true) ** 2

    # Compute mean
    mse = np.mean(squared_error)

    return float(mse)


# =========================
# Example 1
# =========================
y_pred1 = [2, 3]
y_true1 = [1, 1]

print("Example 1 MSE:", mean_squared_error(y_pred1, y_true1))


# =========================
# Example 2
# =========================
y_pred2 = [0, 0, 0]
y_true2 = [0, 0, 0]

print("Example 2 MSE:", mean_squared_error(y_pred2, y_true2))