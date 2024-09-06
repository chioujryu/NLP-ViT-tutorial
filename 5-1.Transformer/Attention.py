import numpy as np

class ScaledDotProductAttention:
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Step 1: MatMul Q and K^T using np.einsum for flexibility
        scores = np.einsum('bqd,bkd->bqk', Q, K)  # shape = (1, 10, 10)

        # Step 2: Scale the scores
        scores = scores / np.sqrt(self.d_k)  # shape = (1, 10, 10)

        # Step 3: Apply Mask (optional)
        if mask is not None:
            scores = scores * mask - 1e9 * (1 - mask)

        # Step 4: SoftMax
        attention_weights = self.softmax(scores) # shape = (1, 10, 10)

        # Step 5: MatMul with V
        output = np.matmul(attention_weights, V) # (1, 10, 10) x (1, 10, 64) = (1, 10, 64)

        return output, attention_weights

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example usage:
d_k = 64  # Dimension of K (and Q)
attention = ScaledDotProductAttention(d_k)

# Sample Q, K, V matrices (batch_size, seq_length, d_k)
Q = np.random.rand(1, 10, d_k)  # Example Query  shape = (1, 10, 64)
K = np.random.rand(1, 10, d_k)  # Example Key    shape = (1, 10, 64)
V = np.random.rand(1, 10, d_k)  # Example Value  shape = (1, 10, 64)

# Forward pass
output, attention_weights = attention.forward(Q, K, V)

print("Output:\n", output)
print("Attention Weights:\n", attention_weights)
