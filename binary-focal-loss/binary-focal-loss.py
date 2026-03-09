import numpy as np

def binary_focal_loss(predictions, targets, alpha=1.0, gamma=2.0, eps=1e-12):
    """
    Compute Binary Focal Loss.

    Parameters
    ----------
    predictions : array-like
        Predicted probabilities (N,), values in (0,1)

    targets : array-like
        Binary targets {0,1}, shape (N,)

    alpha : float
        Balancing factor

    gamma : float
        Focusing parameter

    eps : float
        Numerical stability constant

    Returns
    -------
    float
        Mean focal loss
    """

    # Convert inputs to numpy arrays
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    # Ensure shapes match
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have same shape")

    # Clip probabilities for numerical stability
    predictions = np.clip(predictions, eps, 1 - eps)

    # Compute p_t
    p_t = targets * predictions + (1 - targets) * (1 - predictions)

    # Compute focal loss
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)

    # Return mean loss
    return float(np.mean(loss))


# =========================
# Example 1
# =========================
predictions1 = [0.9]
targets1 = [1]

print("Example 1 Loss:", binary_focal_loss(predictions1, targets1, alpha=1.0, gamma=2.0))


# =========================
# Example 2
# =========================
predictions2 = [0.1]
targets2 = [1]

print("Example 2 Loss:", binary_focal_loss(predictions2, targets2, alpha=1.0, gamma=2.0))