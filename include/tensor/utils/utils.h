#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "tensor/tensor.h"

#define PRINT_LIMIT 3

void print_tensor_shape(Tensor *tensor);
void print_tensor_array_recursive(float *array, int *shape, int ndim, int *indices, int level);
void print_tensor_data(Tensor *tensor);
void print_tensor_grad(Tensor *tensor);
int check_shape_compatibility(Tensor *a, Tensor *b);

#endif