import numpy as np

def label_smoothing_loss(predictions, target, epsilon=0.1, eps=1e-12):
    """
    Compute Cross-Entropy Loss with Label Smoothing.

    Parameters
    ----------
    predictions : array-like
        Predicted probability distribution (K,)
    target : int
        Index of the correct class
    epsilon : float
        Smoothing factor (0 <= epsilon <= 1)
    eps : float
        Numerical stability constant

    Returns
    -------
    float
        Label smoothing loss
    """

    # Convert to numpy array
    predictions = np.asarray(predictions, dtype=float)

    # Number of classes
    K = len(predictions)

    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, eps, 1.0)

    # Build smoothed target distribution
    q = np.full(K, epsilon / K)
    q[target] = (1 - epsilon) + (epsilon / K)

    # Cross-entropy loss
    loss = -np.sum(q * np.log(predictions))

    return float(loss)


# =========================
# Example 1
# =========================
pred1 = [0.9, 0.05, 0.05]
target1 = 0
epsilon1 = 0.1

print("Example 1 Loss:", label_smoothing_loss(pred1, target1, epsilon1))


# =========================
# Example 2
# =========================
pred2 = [0.7, 0.3]
target2 = 0
epsilon2 = 0.2

print("Example 2 Loss:", label_smoothing_loss(pred2, target2, epsilon2))