import numpy as np

# Numerically stable sigmoid
def _sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.1, steps=500):
    
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    N, D = X.shape

    # Initialize parameters
    w = np.zeros(D)
    b = 0.0

    for step in range(steps):

        # Linear combination
        z = np.dot(X, w) + b

        # Prediction
        p = _sigmoid(z)

        # Binary Cross Entropy Loss
        loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

        # Gradients
        dw = (1 / N) * np.dot(X.T, (p - y))
        db = (1 / N) * np.sum(p - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

        # Print progress every 50 steps
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    return w, b


# Example Dataset
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

# Train model
w, b = train_logistic_regression(X, y, lr=0.1, steps=500)

print("\nLearned Parameters:")
print("w:", w)
print("b:", b)


# Prediction function
def predict(X, w, b):
    z = np.dot(X, w) + b
    p = _sigmoid(z)
    return (p >= 0.5).astype(int)


# Check accuracy
preds = predict(X, w, b)
accuracy = np.mean(preds == y)

print("Predictions:", preds)
print("Accuracy:", accuracy)