import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.

    Parameters
    ----------
    y_true : array-like
        Correct class labels (shape: N)

    y_pred : array-like
        Predicted probabilities (shape: N, K)

    Returns
    -------
    float
        Average cross-entropy loss
    """

    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Number of samples
    N = y_true.shape[0]

    # Select predicted probability of the correct class
    correct_probs = y_pred[np.arange(N), y_true]

    # Compute loss
    loss = -np.log(correct_probs)

    # Return mean loss
    return np.mean(loss)


# =========================
# Example 1
# =========================
y_true1 = [0, 1]
y_pred1 = [
    [0.9, 0.1],
    [0.3, 0.7]
]

loss1 = cross_entropy_loss(y_true1, y_pred1)
print("Example 1 Loss:", loss1)


# =========================
# Example 2
# =========================
y_true2 = [2]
y_pred2 = [
    [0.1, 0.1, 0.8]
]

loss2 = cross_entropy_loss(y_true2, y_pred2)
print("Example 2 Loss:", loss2)


# =========================
# Example 3
# =========================
y_true3 = [1, 0, 1]
y_pred3 = [
    [0.2, 0.8],
    [0.6, 0.4],
    [0.49, 0.51]
]

loss3 = cross_entropy_loss(y_true3, y_pred3)
print("Example 3 Loss:", loss3)