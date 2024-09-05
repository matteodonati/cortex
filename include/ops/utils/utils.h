#ifndef OPS_UTILS_H
#define OPS_UTILS_H

#include "tensor/tensor.h"

int check_shape_compatibility(Tensor *a, Tensor *b);
void adjust_indices_for_broadcasting(Tensor *a, Tensor *b, int *a_index, int *b_index, int i);
void compute_reduce_mask_and_divisor(Tensor *tensor, int *axes, int num_axes, int *reduce_mask, int *divisor);
void accumulate_grad(Tensor *self, Tensor *tensor, int *reduce_mask, int divisor, bool is_var, float *mean, bool apply_var);
void backward(Tensor *tensor);
Tensor* one_hot_encode(Tensor *labels, int num_classes);

#endif