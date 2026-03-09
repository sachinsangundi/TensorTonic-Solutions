import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    """
    Compute Contrastive Loss for Siamese Networks.

    Parameters
    ----------
    a : array-like
        First embedding vector(s), shape (D,) or (N,D)

    b : array-like
        Second embedding vector(s), shape (D,) or (N,D)

    y : array-like
        Labels (1 = similar, 0 = dissimilar), shape (N,)

    margin : float
        Margin parameter

    reduction : str
        "mean" or "sum"

    Returns
    -------
    float
        Contrastive loss
    """

    # Convert to numpy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y)

    # Handle single vector input
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Validate labels
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    # Euclidean distance
    diff = a - b
    d = np.sqrt(np.sum(diff**2, axis=1))

    # Loss calculation
    positive_loss = y * (d ** 2)
    negative_loss = (1 - y) * (np.maximum(0, margin - d) ** 2)

    loss = positive_loss + negative_loss

    # Reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")


# =========================
# Example 1
# =========================
a = [1., 0.]
b = [1., 0.]
y = [1]

print("Example 1 Loss:", contrastive_loss(a, b, y, margin=1.0))


# =========================
# Example 2
# =========================
a = [0., 0.]
b = [0.5, 0.]
y = [0]

print("Example 2 Loss:", contrastive_loss(a, b, y, margin=1.0))


# =========================
# Example 3
# =========================
a = [[0.,0.],[1.,1.]]
b = [[0.,0.],[2.,2.]]
y = [1,0]

print("Example 3 Loss:", contrastive_loss(a, b, y, margin=1.0))