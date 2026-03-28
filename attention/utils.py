import numpy as np
from scipy import special


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def add_residual_connection(x, y):
    return x + y


def gelu(x):
    # Matches torch.nn.functional.gelu (exact / erf formulation).
    return 0.5 * x * (1.0 + special.erf(x / np.sqrt(2.0)))


def FFN(x, W1, b1, W2, b2):
    hidden = gelu(x @ W1 + b1)
    return hidden @ W2 + b2


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
