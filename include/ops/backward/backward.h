#ifndef OPS_BACKWARD_H
#define OPS_BACKWARD_H

#include "tensor/tensor.h"

void tensor_add_backward(tensor_t* __restrict__ self);
void tensor_reshape_backward(tensor_t* __restrict__ self);

#endif