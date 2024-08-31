#ifndef NN_CONV2D_PARAMETERS_H
#define NN_CONV2D_PARAMETERS_H

#include "nn/parameters/parameters.h"

typedef struct Conv2DParameters
{
    Parameters base;
    Tensor *weights;
    Tensor *bias;
} Conv2DParameters;

Parameters* conv2d_parameters_create(const char *name, int in_channels, int out_channels, int kernel_size[2], int groups);
Tensor **conv2d_get_params(Parameters *base_params);
void conv2d_parameters_freeze(Parameters *base_params);
void conv2d_parameters_free(Parameters *base_params);

#endif