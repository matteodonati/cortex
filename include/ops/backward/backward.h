#ifndef OPS_BACKWARD_H
#define OPS_BACKWARD_H

#include "tensor/tensor.h"

void tensor_negate_backward(Tensor *self, Tensor *grad);
void tensor_abs_backward(Tensor *self, Tensor *grad);
void tensor_sqrt_backward(Tensor *self, Tensor *grad);
void tensor_exp_backward(Tensor *self, Tensor *grad);
void tensor_add_backward(Tensor *self, Tensor *grad);
void tensor_sub_backward(Tensor *self, Tensor *grad);
void tensor_mul_backward(Tensor *self, Tensor *grad);
void tensor_div_backward(Tensor *self, Tensor *grad);
void tensor_dot_backward(Tensor *self, Tensor *grad);
void tensor_transpose_backward(Tensor *self, Tensor *grad);
void tensor_reshape_backward(Tensor *self, Tensor *grad);

#endif
