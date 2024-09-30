#ifndef OPS_FORWARD_H
#define OPS_FORWARD_H

#include "tensor/tensor.h"

tensor_t* tensor_add(const tensor_t* __restrict__ a, const tensor_t* __restrict__ b);
tensor_t* tensor_reshape(const tensor_t* __restrict__ tensor, const size_t* __restrict__ new_shape, size_t new_ndim);

#endif