#ifndef NN_LAYER_CONV2D_H
#define NN_LAYER_CONV2D_H

#include "nn/layers/layer.h"

typedef struct Conv2D
{
    Layer base;
    int kernel_size[2];
    int stride[2];
    int padding[2];
    int in_channels;
    int out_channels;
    int groups;
} Conv2D;

Layer *conv2d_create(const char *name, int in_channels, int out_channels, int kernel_size[2], int stride[2], int padding[2]);
Tensor* im2col(Tensor *input, int kernel_height, int kernel_width, int stride_height, int stride_width, int pad_height, int pad_width);
void col2im(Tensor *self);
Tensor *conv2d_forward(Layer *self, Tensor *x);
void conv2d_free(Layer *self);

#endif