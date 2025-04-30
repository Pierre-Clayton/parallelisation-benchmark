#pragma once
#include <cmath>

// attention_impl_cpp pour float et double
void attention_impl_cpp(int N, int D, int DV,
                       const float* Q, const float* K, const float* V,
                       float* out,
                       int block_size,
                       int version);
void attention_impl_cpp(int N, int D, int DV,
                       const double* Q, const double* K, const double* V,
                       double* out,
                       int block_size,
                       int version);