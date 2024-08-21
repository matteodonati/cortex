#ifndef NN_DENSE_H
#define NN_DENSE_H

#include "nn/layer.h"

typedef struct Dense 
{
    Layer base;
    int input_dim;
    int output_dim;
} Dense;

Layer* dense_create(const char *name, int input_dim, int output_dim);
Tensor* dense_forward(Layer *self, Tensor *input);
Tensor** dense_get_params(Layer *self, int *num_params);
void dense_freeze_params(Layer *self);
void dense_free(Layer *self);

#endif
