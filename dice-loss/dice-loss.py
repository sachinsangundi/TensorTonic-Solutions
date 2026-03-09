import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.

    Parameters
    ----------
    p : array-like
        Predicted probabilities (N,) or (H,W)

    y : array-like
        Ground truth binary mask {0,1}

    eps : float
        Small value for numerical stability

    Returns
    -------
    float
        Dice Loss
    """

    # Convert inputs to float arrays
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Ensure same shape
    if p.shape != y.shape:
        raise ValueError("p and y must have the same shape")

    # Flatten arrays (works for 1D or 2D)
    p = p.reshape(-1)
    y = y.reshape(-1)

    # Intersection
    intersection = np.sum(p * y)

    # Sum of masks
    sum_p = np.sum(p)
    sum_y = np.sum(y)

    # Dice coefficient
    dice = (2 * intersection + eps) / (sum_p + sum_y + eps)

    # Dice loss
    loss = 1 - dice

    return loss


# =========================
# Example 1
# =========================
p1 = [0.9, 0.7, 0.1, 0.0]
y1 = [1, 1, 0, 0]

print("Example 1 Loss:", dice_loss(p1, y1))


# =========================
# Example 2
# =========================
p2 = [1.0, 1.0, 0.0, 0.0]
y2 = [1, 1, 0, 0]

print("Example 2 Loss:", dice_loss(p2, y2))


# =========================
# Example 3
# =========================
p3 = [1.0, 1.0]
y3 = [0, 0]

print("Example 3 Loss:", dice_loss(p3, y3))