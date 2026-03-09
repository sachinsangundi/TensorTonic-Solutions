import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.

    Parameters
    ----------
    Z1 : array-like
        First embedding batch (N, D)

    Z2 : array-like
        Second embedding batch (N, D)

    temperature : float
        Temperature parameter (tau)

    Returns
    -------
    float
        Mean InfoNCE loss
    """

    # Convert to numpy arrays
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    # Validate shapes
    if Z1.shape != Z2.shape:
        raise ValueError("Z1 and Z2 must have the same shape")

    # Compute similarity matrix
    S = np.dot(Z1, Z2.T) / temperature

    # Numerical stability (softmax trick)
    S = S - np.max(S, axis=1, keepdims=True)

    # Exponentiate
    exp_S = np.exp(S)

    # Denominator: sum over rows
    denom = np.sum(exp_S, axis=1)

    # Numerator: diagonal elements
    num = np.diag(exp_S)

    # Compute loss
    loss = -np.log(num / denom)

    # Mean loss
    return float(np.mean(loss))


# =========================
# Example 1
# =========================
Z1 = [[1,0],[0,1]]
Z2 = [[1,0],[0,1]]

print("Example 1 Loss:", info_nce_loss(Z1, Z2, temperature=0.1))


# =========================
# Example 2
# =========================
Z1 = [[1,0],[0,1]]
Z2 = [[0,1],[1,0]]

print("Example 2 Loss:", info_nce_loss(Z1, Z2, temperature=0.1))


# =========================
# Example 3
# =========================
Z1 = [[1,0],[0,1]]
Z2 = [[1,0],[0,1]]

print("Example 3 Loss:", info_nce_loss(Z1, Z2, temperature=1.0))