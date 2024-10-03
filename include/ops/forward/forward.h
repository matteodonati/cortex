#ifndef OPS_FORWARD_H
#define OPS_FORWARD_H

#include "tensor/tensor.h"

tensor_t* tensor_add(const tensor_t* a, const tensor_t* b);
tensor_t* tensor_reshape(const tensor_t* tensor, const size_t* new_shape, size_t new_ndim);

#endif