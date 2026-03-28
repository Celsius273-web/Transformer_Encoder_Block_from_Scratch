import numpy as np
import torch
import torch.nn.functional as F
from attention import scaled_dot_product_attention
#Claude Generated Test to compare my implementation of attention mechanism vs PyTorch's implementation of attention mechanism
def test_against_pytorch():
    Q = np.random.randn(4, 8)
    K = np.random.randn(4, 8)
    V = np.random.randn(4, 8)

    your_output = scaled_dot_product_attention(Q, K, V)

    Q_t = torch.tensor(Q, dtype=torch.float32)
    K_t = torch.tensor(K, dtype=torch.float32)
    V_t = torch.tensor(V, dtype=torch.float32)
    pytorch_output = F.scaled_dot_product_attention(Q_t, K_t, V_t).numpy()

    np.testing.assert_allclose(your_output, pytorch_output, atol=1e-5)
    print("Test passed")

test_against_pytorch()