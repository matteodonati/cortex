#ifndef NN_ACTIVATIONS_FORWARD_H
#define NN_ACTIVATIONS_FORWARD_H

#include "tensor/tensor.h"

#define M_PI 3.14159265358979323846

Tensor* relu_f(Tensor *tensor);
Tensor* sigmoid_f(Tensor *tensor);
Tensor* tanh_f(Tensor *tensor);
Tensor* leaky_relu_f(Tensor *tensor, float alpha);
Tensor* elu_f(Tensor *tensor, float alpha);
Tensor* gelu_f(Tensor *tensor);
Tensor* softmax_f(Tensor *tensor, int axis);

#endif