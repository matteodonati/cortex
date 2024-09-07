#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nn/parameters/dense.h"

Parameters* dense_parameters_create(const char *name, int input_dim, int output_dim) 
{
    DenseParameters *params = (DenseParameters *)malloc(sizeof(DenseParameters));
    
    char weights_name[256];
    snprintf(weights_name, sizeof(weights_name), "%s.weight", name);

    char bias_name[256];
    snprintf(bias_name, sizeof(bias_name), "%s.bias", name);

    float limit = sqrtf(1.0f / input_dim);

    int weights_ndim = 2;
    int weights_shape[] = {output_dim, input_dim};
    params->weights = tensor_rand(weights_name, weights_shape, weights_ndim, limit);
    
    int bias_ndim = 1;
    int bias_shape[] = {output_dim};
    params->bias = tensor_rand(bias_name, bias_shape, bias_ndim, limit);

    params->base.get_params = &dense_get_params;
    params->base.num_params = 2;
    params->base.freeze_params = &dense_parameters_freeze;
    params->base.free = &dense_parameters_free;

    return (Parameters *)params;
}

Tensor **dense_get_params(Parameters *base_params) 
{
    DenseParameters *params = (DenseParameters *)base_params;
    Tensor **param_array = (Tensor **)malloc(2 * sizeof(Tensor *));
    param_array[0] = params->weights;
    param_array[1] = params->bias;
    return param_array;
}

void dense_parameters_freeze(Parameters *base_params) 
{
    DenseParameters *params = (DenseParameters *)base_params;
    if (params->weights) 
    {
        params->weights->frozen = true;
    }
    if (params->bias) 
    {
        params->bias->frozen = true;
    }
}

void dense_parameters_free(Parameters *base_params) 
{
    DenseParameters *params = (DenseParameters *)base_params;
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