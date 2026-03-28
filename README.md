Project: Transformer Encoder Block from Scratch
A NumPy implementation of a transformer encoder block validated against PyTorch with a 0.0000003 mean absolute error.
View the full notebook with outputs here: https://htmlpreview.github.io/?https://github.com/Celsius273-web/Transformer_Encoder_Block_from_Scratch/blob/main/notebook/test_encoder_block.html

What this implements

Scaled dot-product attention
Multi-head self-attention with Q, K, V projections
Residual connections and layer normalization
Feed-forward network with GELU activation
Numerical validation against PyTorch's TransformerEncoderLayer

Result
Max absolute difference of my implementation vs PyTorch Encoder Block: 3e-7
This is well within the 1e-5 goal I set, proving the implementation is numerically correct. Feel free to run yourself!

How to run
git clone https://github.com/YOURUSERNAME/YOURREPONAME
cd YOURREPONAME
pip install -r requirements.txt
jupyter notebook notebook/test_encoder_block.ipynb
