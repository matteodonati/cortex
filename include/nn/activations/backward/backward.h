#ifndef NN_ACTIVATIONS_BACKWARD_H
#define NN_ACTIVATIONS_BACKWARD_H

#include "tensor/tensor.h"

void relu_backward(Tensor *self, float *grad);
void sigmoid_backward(Tensor *self, float *grad);
void tanh_backward(Tensor *self, float *grad);
void leaky_relu_backward(Tensor *self, float *grad);
void elu_backward(Tensor *self, float *grad);
void gelu_backward(Tensor *self, float *grad);
void softmax_backward(Tensor *self, float *grad);

#endif