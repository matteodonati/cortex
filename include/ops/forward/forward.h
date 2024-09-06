#ifndef OPS_FORWARD_H
#define OPS_FORWARD_H

#include "tensor/tensor.h"

Tensor* tensor_negate(Tensor *tensor);
Tensor* tensor_abs(Tensor *tensor);
Tensor* tensor_sqrt(Tensor *tensor);
Tensor* tensor_exp(Tensor *tensor);
Tensor* tensor_log(Tensor *tensor);
Tensor* tensor_add(Tensor *a, Tensor *b);
Tensor* tensor_sub(Tensor *a, Tensor *b);
Tensor* tensor_mul(Tensor *a, Tensor *b);
Tensor* tensor_div(Tensor *a, Tensor *b);
Tensor* tensor_scalar_add(Tensor *tensor, float scalar);
Tensor* tensor_scalar_mul(Tensor *tensor, float scalar);
Tensor* tensor_matmul(Tensor *a, Tensor *b);
Tensor* tensor_reshape(Tensor *tensor, int *new_shape, int new_ndim);
Tensor* tensor_transpose(Tensor *tensor, int *axes);
Tensor* tensor_max(Tensor *tensor, int axis);
Tensor* tensor_min(Tensor *tensor, int axis);
Tensor* tensor_argmax(Tensor *tensor, int axis);
Tensor* tensor_argmin(Tensor *tensor, int axis);
Tensor* tensor_sum(Tensor *tensor, int *axes, int num_axes);
Tensor* tensor_mean(Tensor *tensor, int *axes, int num_axes);
Tensor* tensor_var(Tensor *tensor, int *axes, int num_axes, bool unbiased);
Tensor* tensor_cat(Tensor *a, Tensor *b, int axis);
Tensor* im2col(Tensor *input, int kernel_height, int kernel_width, int stride_height, int stride_width, int pad_height, int pad_width);
Tensor* tensor_normalize2d(Tensor *x, bool is_training, Tensor *mean, Tensor *var, int *axes, int num_axes, float epsilon);

#endif
