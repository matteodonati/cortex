#ifndef NN_LAYER_MAXPOOL2D_H
#define NN_LAYER_MAXPOOL2D_H

#include "nn/layers/layer.h"

typedef struct MaxPool2D 
{
    Layer base;
    int kernel_size[2];
    int stride[2];
    int padding[2];
} MaxPool2D;

Layer* maxpool2d_create(const char *name, int kernel_size[2], int stride[2], int padding[2]);
Tensor* maxpool2d_forward(Layer *self, Tensor *x);
void maxpool2d_free(Layer *self);

#endif