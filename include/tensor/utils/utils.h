#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "tensor/tensor.h"

#define PRINT_LIMIT 3

void print_tensor_array_recursive(float *data, int *shape, int ndim, int current_dim, int *indices);
void print_tensor(Tensor *tensor, const char *name);

#endif