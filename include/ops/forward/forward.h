#ifndef OPS_FORWARD_H
#define OPS_FORWARD_H

#include "tensor/tensor.h"

Tensor* tensor_negate(Tensor *tensor);
Tensor* tensor_abs(Tensor *tensor);
Tensor* tensor_sqrt(Tensor *tensor);
Tensor* tensor_exp(Tensor *tensor);
Tensor* tensor_add(Tensor *a, Tensor *b);
Tensor* tensor_sub(Tensor *a, Tensor *b);
Tensor* tensor_mul(Tensor *a, Tensor *b);
Tensor* tensor_div(Tensor *a, Tensor *b);
Tensor* tensor_matmul(Tensor *a, Tensor *b);
Tensor* tensor_reshape(Tensor *tensor, int *new_shape, int new_ndim);
Tensor* tensor_transpose(Tensor *tensor, int *axes);

#endif
