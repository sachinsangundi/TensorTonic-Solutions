import numpy as np

def sigmoid(x):
    """
    Computes the sigmoid activation function.
    
    Parameters:
    x : scalar, list, or numpy array
    
    Returns:
    numpy array (float)
    """
    x = np.array(x, dtype=float)   # convert input to NumPy array
    return 1 / (1 + np.exp(-x))


# -------- Example Tests --------

# Example 1
x1 = [0, 2, -2]
print("Input:", x1)
print("Output:", sigmoid(x1))

# Example 2
x2 = 0
print("\nInput:", x2)
print("Output:", sigmoid(x2))

# Example 3
x3 = [[-1, 0], [1, 2]]
print("\nInput:\n", x3)
print("Output:\n", sigmoid(x3))