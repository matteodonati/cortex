#ifndef NN_DENSE_H
#define NN_DENSE_H

#include "layer.h"

typedef struct Dense 
{
    Layer base;
    int input_dim;
    int output_dim;
} Dense;

Layer* dense_create(int input_dim, int output_dim);
Tensor* dense_forward(Layer *self, Tensor *input);
void dense_backward(Layer *self, Tensor *grad_output);
void dense_update_params(Layer *self, Optimizer *optimizer);
void dense_free(Layer *self);

#endif
