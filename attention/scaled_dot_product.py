import numpy as np

from . import utils

def scaled_dot_product_attention(Q, K, V, return_weights=False):
    d_k = K.shape[1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = utils.softmax(scores, axis=-1)
    output = weights @ V
    if return_weights:
        return output, weights
    return output
