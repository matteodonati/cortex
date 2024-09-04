#ifndef NN_LAYER_AVGPOOL2D_H
#define NN_LAYER_AVGPOOL2D_H

#include "nn/layers/layer.h"

typedef struct AvgPool2D 
{
    Layer base;
    int kernel_size[2];
    int stride[2];
    int padding[2];
} AvgPool2D;

Layer* avgpool2d_create(const char *name, int kernel_size[2], int stride[2], int padding[2]);
Tensor* avgpool2d_forward(Layer *self, Tensor *x);
void avgpool2d_free(Layer *self);

#endif