#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nn/parameters/conv2d.h"

Parameters* conv2d_parameters_create(const char *name, int in_channels, int out_channels, int kernel_size[2], int groups) 
{
    Conv2DParameters *params = (Conv2DParameters *)malloc(sizeof(Conv2DParameters));

    float limit = sqrtf(groups / (float)(in_channels * kernel_size[0] * kernel_size[1]));

    char weights_name[256];
    snprintf(weights_name, sizeof(weights_name), "%s.weight", name);
    params->weights = tensor_rand(weights_name, (int[]){out_channels, in_channels / groups, kernel_size[0], kernel_size[1]}, 4, limit);

    char bias_name[256];
    snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
    params->bias = tensor_rand(bias_name, (int[]){out_channels}, 1, limit);

    params->base.get_params = &conv2d_get_params;
    params->base.num_params = 2;
    params->base.freeze_params = &conv2d_parameters_freeze;
    params->base.free = &conv2d_parameters_free;

    return (Parameters *)params;
}

Tensor **conv2d_get_params(Parameters *base_params) 
{
    Conv2DParameters *params = (Conv2DParameters *)base_params;
    Tensor **param_array = (Tensor **)malloc(2 * sizeof(Tensor *));
    param_array[0] = params->weights;
    param_array[1] = params->bias;
    return param_array;
}

void conv2d_parameters_freeze(Parameters *base_params) 
{
    Conv2DParameters *params = (Conv2DParameters *)base_params;
    if (params->weights) 
    {
        params->weights->frozen = true;
    }
    if (params->bias) 
    {
        params->bias->frozen = true;
    }
}

void conv2d_parameters_free(Parameters *base_params) 
{
    Conv2DParameters *params = (Conv2DParameters *)base_params;
    if (params->weights) 
    {
        tensor_free(params->weights);
    }
    if (params->bias) 
    {
        tensor_free(params->bias);
    }
    free(params);
}