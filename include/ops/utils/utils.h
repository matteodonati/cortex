#ifndef OPS_UTILS_H
#define OPS_UTILS_H

#include "tensor/tensor.h"

int check_shape_compatibility(Tensor *a, Tensor *b);
void adjust_indices_for_broadcasting(Tensor *a, Tensor *b, int *a_index, int *b_index, int i);
void tensor_backward(Tensor *tensor);
Tensor* one_hot_encode(Tensor *labels, int num_classes);

#endif