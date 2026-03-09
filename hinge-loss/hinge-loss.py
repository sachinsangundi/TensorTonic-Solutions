import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean"):
    """
    Compute Hinge Loss for binary SVM.

    Parameters
    ----------
    y_true : array-like
        True labels {-1, +1}, shape (N,)

    y_score : array-like
        Predicted scores, shape (N,)

    margin : float
        Margin parameter (default = 1)

    reduction : str
        "mean" or "sum"

    Returns
    -------
    float
        Hinge loss
    """

    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    # Validate label values
    if not np.all((y_true == 1) | (y_true == -1)):
        raise ValueError("y_true must contain only -1 or +1")

    # Compute hinge loss
    losses = np.maximum(0, margin - y_true * y_score)

    # Reduction
    if reduction == "mean":
        return float(np.mean(losses))
    elif reduction == "sum":
        return float(np.sum(losses))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")


# =========================
# Example 1
# =========================
y_true1 = [1, 1, -1]
y_score1 = [2, 0, 0]

print("Example 1 Loss:", hinge_loss(y_true1, y_score1))


# =========================
# Example 2
# =========================
y_true2 = [-1, 1]
y_score2 = [-3, 0.5]

print("Example 2 Loss:", hinge_loss(y_true2, y_score2))