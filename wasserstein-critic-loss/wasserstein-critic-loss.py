import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.

    Parameters
    ----------
    real_scores : array-like
        Critic outputs for real samples

    fake_scores : array-like
        Critic outputs for fake samples

    Returns
    -------
    float
        Wasserstein critic loss
    """

    # Convert inputs to NumPy arrays
    real_scores = np.asarray(real_scores, dtype=float)
    fake_scores = np.asarray(fake_scores, dtype=float)

    # Compute means
    mean_real = np.mean(real_scores)
    mean_fake = np.mean(fake_scores)

    # Compute Wasserstein loss
    loss = mean_fake - mean_real

    return float(loss)


# =========================
# Example 1
# =========================
real_scores = [2.0, 1.5, 3.0]
fake_scores = [-1.0, 0.0, 0.5]

print("Example 1 Loss:", wasserstein_critic_loss(real_scores, fake_scores))


# =========================
# Example 2
# =========================
real_scores = [1.0, 2.0, 3.0]
fake_scores = [2.0, 2.0, 2.0]

print("Example 2 Loss:", wasserstein_critic_loss(real_scores, fake_scores))


# =========================
# Example 3
# =========================
real_scores = [0.0, 0.0]
fake_scores = [1.0, 2.0, 3.0]

print("Example 3 Loss:", wasserstein_critic_loss(real_scores, fake_scores))