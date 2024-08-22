#ifndef NN_ACTIVATIONS_FORWARD_H
#define NN_ACTIVATIONS_FORWARD_H

#include "tensor/tensor.h"

Tensor* relu_f(Tensor *input);
Tensor* sigmoid_f(Tensor *input);
Tensor* tanh_f(Tensor *input);
Tensor* softmax_f(Tensor *input, int axis);

#endif