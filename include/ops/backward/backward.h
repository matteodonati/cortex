#ifndef OPS_BACKWARD_H
#define OPS_BACKWARD_H

#include "tensor/tensor.h"

void tensor_negate_backward(Tensor *self);
void tensor_abs_backward(Tensor *self);
void tensor_sqrt_backward(Tensor *self);
void tensor_exp_backward(Tensor *self);
void tensor_log_backward(Tensor *self);
void tensor_add_backward(Tensor *self);
void tensor_sub_backward(Tensor *self);
void tensor_mul_backward(Tensor *self);
void tensor_div_backward(Tensor *self);
void tensor_scalar_add_backward(Tensor *self);
void tensor_scalar_mul_backward(Tensor *self);
void tensor_matmul_backward(Tensor *self);
void tensor_reshape_backward(Tensor *self);
void tensor_transpose_backward(Tensor *self);
void tensor_max_min_backward(Tensor *self);
void tensor_sum_backward(Tensor *self);
void tensor_mean_backward(Tensor *self);
void tensor_var_backward(Tensor *self);
void tensor_cat_backward(Tensor *self);
void col2im(Tensor *self);
void tensor_normalize2d_backward(Tensor *self);

#endif
