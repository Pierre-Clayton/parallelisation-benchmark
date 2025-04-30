import time
import numpy as np
import attention

def numpy_attention(Q, K, V):
    # scaled dot-product
    D = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(D)
    P = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    P /= np.sum(P, axis=1, keepdims=True)
    return P @ V

if __name__ == '__main__':
    for N in [128, 256, 512]:
        for D in [64, 128]:
            Q = np.random.rand(N, D).astype(np.float32)
            K = np.random.rand(N, D).astype(np.float32)
            V = np.random.rand(N, D).astype(np.float32)
            # Warm-up
            _ = numpy_attention(Q, K, V)
            _ = attention.attention(Q, K, V, version=1)

            # numpy
            t0 = time.time()
            _ = numpy_attention(Q, K, V)
            tn = time.time() - t0
            # optimized
            t0 = time.time()
            _ = attention.attention(Q, K, V, version=1)
            to = time.time() - t0

            print(f"N={N}, D={D}: numpy {tn*1000:.2f} ms | optimized {to*1000:.2f} ms | speedup {tn/to:.2f}x")