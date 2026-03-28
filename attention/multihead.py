import numpy as np

from .scaled_dot_product import scaled_dot_product_attention


def multihead_attention(Q, K, V, num_heads, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, return_attention_weights=False):

    #Multi-head self- or cross-attention.
    #This implementation additionally requires projection weights and biases to fairly compare with PyTorch's implementation.
    #d_model is the dimension of the model.
    d_model = Q.shape[-1]
    assert d_model % num_heads == 0
    d_k = d_model // num_heads
    #Project Q, K, V to d_k dimensions.
    Q = Q @ W_q + b_q
    K = K @ W_k + b_k
    V = V @ W_v + b_v
    #Reshape Q, K, V to num_heads x d_k dimensions.
    seq_len = Q.shape[0]
    Qh = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    Kh = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    Vh = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    #Compute attention for each head.
    head_outs = []
    head_weights = []
    for h in range(num_heads):
        if return_attention_weights:
            out_h, w_h = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], return_weights=True)
            head_weights.append(w_h)
        else:
            out_h = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h])
        head_outs.append(out_h)
    #Concatenate the heads.
    concat = np.concatenate(head_outs, axis=-1)
    output = concat @ W_o + b_o

    if return_attention_weights:
        attn_weights = np.stack(head_weights, axis=0)
        return output, attn_weights
    return output
