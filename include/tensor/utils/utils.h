#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "tensor/tensor.h"

#define PRINT_LIMIT 3

void tensor_set_slice(Tensor *dest, Tensor *src, int slice_index);
void print_tensor_array_recursive(float *data, int *shape, int ndim, int current_dim, int *indices);
void print_tensor(Tensor *tensor, const char *name);

#endif