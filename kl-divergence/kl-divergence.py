import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q)

    Parameters
    ----------
    p : array-like
        Reference probability distribution (N,)
    q : array-like
        Approximation distribution (N,)
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    float
        KL divergence value
    """

    # Convert inputs to numpy arrays
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Validate shapes
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    # Add epsilon to avoid log(0)
    q = q + eps

    # Compute KL divergence
    kl = np.sum(np.where(p > 0, p * np.log(p / q), 0.0))

    return float(kl)


# =========================
# Example 1
# =========================
p1 = [0.4, 0.6]
q1 = [0.5, 0.5]

print("Example 1 KL:", kl_divergence(p1, q1))


# =========================
# Example 2
# =========================
p2 = [0.3, 0.7]
q2 = [0.3, 0.7]

print("Example 2 KL:", kl_divergence(p2, q2))


# =========================
# Example 3
# =========================
p3 = [0.9, 0.1]
q3 = [0.5, 0.5]

print("Example 3 KL:", kl_divergence(p3, q3))