import numpy as np

from . import utils
from .multihead import multihead_attention

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        scale = 0.02

        self.W_q = (np.random.randn(d_model, d_model) * scale).astype(np.float32) #Query projection weights.
        self.W_k = (np.random.randn(d_model, d_model) * scale).astype(np.float32) #Key projection weights.
        self.W_v = (np.random.randn(d_model, d_model) * scale).astype(np.float32) #Value projection weights.
        self.W_o = (np.random.randn(d_model, d_model) * scale).astype(np.float32) #Output projection weights.
        self.b_q = np.zeros(d_model, dtype=np.float32) #Output projection biases.
        self.b_k = np.zeros(d_model, dtype=np.float32) #Output projection biases.
        self.b_v = np.zeros(d_model, dtype=np.float32) #Output projection biases.
        self.b_o = np.zeros(d_model, dtype=np.float32) #Output projection biases.

        self.ln1_gamma = np.ones(d_model, dtype=np.float32) #Layer normalization gamma.
        self.ln1_beta = np.zeros(d_model, dtype=np.float32) #Layer normalization beta.
        self.ln2_gamma = np.ones(d_model, dtype=np.float32) #Layer normalization gamma.
        self.ln2_beta = np.zeros(d_model, dtype=np.float32) #Layer normalization beta.

        self.W1 = (np.random.randn(d_model, d_ff) * scale).astype(np.float32) #First linear weights.
        self.b1 = np.zeros(d_ff, dtype=np.float32) #First linear biases.
        self.W2 = (np.random.randn(d_ff, d_model) * scale).astype(np.float32) #Second linear weights.
        self.b2 = np.zeros(d_model, dtype=np.float32) #Second linear biases.

        self.last_attention_weights = None

    def forward(self, x, return_attention_weights=False):
        #X is the input to the encoder block and it is the same for the query, key, and value.
        attn_out = multihead_attention(x, x, x, self.num_heads, self.W_q, self.W_k, self.W_v, self.W_o, self.b_q, self.b_k, self.b_v, self.b_o, return_attention_weights=return_attention_weights)
        if return_attention_weights:
            attn_out, attn_w = attn_out
            self.last_attention_weights = attn_w
        # Gamma and beta are the scale and shift parameters for the layer normalization.
        x = utils.layer_norm(utils.add_residual_connection(x, attn_out), self.ln1_gamma, self.ln1_beta)
        ffn_out = utils.FFN(x, self.W1, self.b1, self.W2, self.b2)
        x = utils.layer_norm(utils.add_residual_connection(x, ffn_out), self.ln2_gamma, self.ln2_beta)
        return x
