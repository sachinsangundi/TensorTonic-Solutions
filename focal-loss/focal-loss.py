import numpy as np

def focal_loss(p, y, gamma=2.0, eps=1e-12):
    """
    Compute Binary Focal Loss.

    Parameters
    ----------
    p : np.ndarray
        Predicted probabilities, shape (N,)
    y : np.ndarray
        True binary labels {0,1}, shape (N,)
    gamma : float
        Focusing parameter (gamma >= 0)
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    float
        Mean focal loss
    """

    # Convert inputs to numpy arrays
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Validate shapes
    if p.shape != y.shape:
        raise ValueError("p and y must have the same shape")

    # Numerical stability
    p = np.clip(p, eps, 1 - eps)

    # Compute focal loss components
    pos_loss = -y * ((1 - p) ** gamma) * np.log(p)
    neg_loss = -(1 - y) * (p ** gamma) * np.log(1 - p)

    # Total loss
    loss = pos_loss + neg_loss

    return float(np.mean(loss))


# =========================
# Example 1
# =========================
p1 = np.array([0.9, 0.2, 0.7, 0.1])
y1 = np.array([1, 0, 1, 0])

print("Example 1 Loss:", focal_loss(p1, y1, gamma=2.0))


# =========================
# Example 2
# =========================
p2 = np.array([0.5, 0.5, 0.5, 0.5])
y2 = np.array([1, 0, 1, 0])

print("Example 2 Loss:", focal_loss(p2, y2, gamma=2.0))


# =========================
# Example 3 (Cross-Entropy case)
# =========================
print("Example 3 Loss:", focal_loss(p1, y1, gamma=0.0))