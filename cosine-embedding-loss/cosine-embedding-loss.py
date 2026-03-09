import numpy as np

def cosine_embedding_loss(x1, x2, label, margin=0.0):
    """
    Compute Cosine Embedding Loss.

    Parameters
    ----------
    x1 : array-like
        First vector
    x2 : array-like
        Second vector
    label : int
        +1 for similar, -1 for dissimilar
    margin : float
        Margin parameter (>= 0)

    Returns
    -------
    float
        Cosine embedding loss
    """

    # Convert inputs to numpy arrays
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # Compute cosine similarity
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    cosine_sim = dot_product / (norm_x1 * norm_x2)

    # Compute loss
    if label == 1:
        loss = 1 - cosine_sim
    elif label == -1:
        loss = max(0.0, cosine_sim - margin)
    else:
        raise ValueError("label must be 1 or -1")

    return float(loss)


# =========================
# Example 1
# =========================
x1 = [1, 0, 0]
x2 = [1, 0, 0]
label = 1

print("Example 1 Loss:", cosine_embedding_loss(x1, x2, label))


# =========================
# Example 2
# =========================
x1 = [1, 0, 0]
x2 = [0, 1, 0]
label = 1

print("Example 2 Loss:", cosine_embedding_loss(x1, x2, label))