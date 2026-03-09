import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    
    # Convert to numpy arrays
    anchor = np.array(anchor, dtype=float)
    positive = np.array(positive, dtype=float)
    negative = np.array(negative, dtype=float)

    # Handle single vector input
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
        positive = positive.reshape(1, -1)
        negative = negative.reshape(1, -1)

    # Squared Euclidean distances
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)

    # Triplet loss
    loss = np.maximum(0, d_ap - d_an + margin)

    # Mean loss
    return np.mean(loss)