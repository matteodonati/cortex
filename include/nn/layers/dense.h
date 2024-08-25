#ifndef NN_LAYER_DENSE_H
#define NN_LAYER_DENSE_H

#include "nn/layers/layer.h"

typedef struct Dense 
{
    Layer base;
    int input_dim;
    int output_dim;
} Dense;

Layer* dense_create(const char *name, int input_dim, int output_dim);
Tensor* dense_forward(Layer *self, Tensor *x);
void dense_free(Layer *self);

#endif
