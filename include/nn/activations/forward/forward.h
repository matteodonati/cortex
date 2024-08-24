#ifndef NN_ACTIVATIONS_FORWARD_H
#define NN_ACTIVATIONS_FORWARD_H

#include "tensor/tensor.h"

#define M_PI 3.14159265358979323846

Tensor* relu_f(Tensor *input);
Tensor* sigmoid_f(Tensor *input);
Tensor* tanh_f(Tensor *input);
Tensor* leaky_relu_f(Tensor *input, float alpha);
Tensor* elu_f(Tensor *input, float alpha);
Tensor* gelu_f(Tensor *input);
Tensor* softmax_f(Tensor *input, int axis);

#endif