# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

cdef extern from "attention_impl.h":
    void attention_impl_cpp(int N, int D, int DV,
                            const float* Q, const float* K, const float* V,
                            float* out,
                            int block_size,
                            int version)
    void attention_impl_cpp(int N, int D, int DV,
                            const double* Q, const double* K, const double* V,
                            double* out,
                            int block_size,
                            int version)

cpdef np.ndarray attention(np.ndarray Q, np.ndarray K, np.ndarray V,
                           int block_size=16, int version=0):
    """Compute scaled dot-product attention: softmax(QK^T/sqrt(D))·V"""
    cdef int N = Q.shape[0]
    cdef int D = Q.shape[1]
    cdef int DV = V.shape[1]
    assert Q.shape[1] == K.shape[1] == V.shape[1]
    assert Q.dtype == K.dtype == V.dtype
    cdef np.ndarray out = np.zeros((N, DV), dtype=Q.dtype)

    if Q.dtype == np.float32:
        attention_impl_cpp(N, D, DV,
                           <const float*>Q.data, <const float*>K.data, <const float*>V.data,
                           <float*>out.data,
                           block_size, version)
    elif Q.dtype == np.float64:
        attention_impl_cpp(N, D, DV,
                           <const double*>Q.data, <const double*>K.data, <const double*>V.data,
                           <double*>out.data,
                           block_size, version)
    else:
        raise NotImplementedError("dtype non supporté")
    return out