#ifndef NN_ACTIVATIONS_BACKWARD_H
#define NN_ACTIVATIONS_BACKWARD_H

#include "tensor/tensor.h"

void relu_backward(Tensor *self);
void sigmoid_backward(Tensor *self);
void tanh_backward(Tensor *self);
void leaky_relu_backward(Tensor *self);
void elu_backward(Tensor *self);
void gelu_backward(Tensor *self);
void softmax_backward(Tensor *self);

#endif