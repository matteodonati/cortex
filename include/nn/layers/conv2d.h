#ifndef NN_LAYER_CONV2D_H
#define NN_LAYER_CONV2D_H

#include "nn/layers/layer.h"

#define PADDING_MODE_ZEROS 0

typedef struct Conv2D
{
    Layer base;
    int in_channels;
    int out_channels;
    int kernel_size[2];
    int stride[2];
    int padding[2];
    int groups;
    int padding_mode;
} Conv2D;

Layer *conv2d_create(const char *name, int in_channels, int out_channels, int kernel_size[2], int stride[2], int padding[2], int groups, int padding_mode, int bias);
Tensor *conv2d_forward(Layer *self, Tensor *x);
void conv2d_free(Layer *self);

#endif