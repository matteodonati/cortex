#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "tensor/tensor.h"

#define PRINT_LIMIT 3

void print_tensor_shape(Tensor *tensor);
void print_tensor_data_recursive(float *data, int *shape, int ndim, int *indices, int level);
void print_tensor_data(Tensor *tensor);

#endif