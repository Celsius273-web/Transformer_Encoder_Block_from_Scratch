Transformer Encoder Block from Scratch:

A NumPy implementation of a transformer encoder block validated against PyTorch with a 0.000000383 mean absolute error.
View the full notebook with outputs here: https://htmlpreview.github.io/?https://github.com/Celsius273-web/Transformer_Encoder_Block_from_Scratch/blob/main/notebook/test_encoder_block.html

What this implements

Scaled dot-product attention

Multi-head self-attention with Q, K, V projections

Residual connections and layer normalization

Feed-forward network with GELU activation

Numerical validation against PyTorch's TransformerEncoderLayer

Result:

Max absolute difference: 3.8308632221983885e-07 of my implementation vs PyTorch Encoder Block
This is well within the 1e-5 goal I set, proving the implementation is numerically correct. Feel free to run yourself!
Heat Map using MatplotLib showing how the encoder maps relationships.


<img width="590" height="490" alt="image" src="https://github.com/user-attachments/assets/008234eb-0ab3-4cf2-8974-ce4bb4646984" />


How to run
git clone https://github.com/Celsius273-web/Transformer_Encoder_Block_from_Scratch

cd Transformer_Encoder_Block_from_Scratch

Create Python venv - best to use python 3.11

pip install -r requirements.txt

jupyter notebook notebook/test_encoder_block.ipynb
