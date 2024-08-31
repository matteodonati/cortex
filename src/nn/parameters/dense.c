#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nn/parameters/dense.h"

Parameters* dense_parameters_create(const char *name, int input_dim, int output_dim) 
{
    DenseParameters *params = (DenseParameters *)malloc(sizeof(DenseParameters));
    
    float limit = sqrtf(1.0f / input_dim);
    
    char weights_name[256];
    snprintf(weights_name, sizeof(weights_name), "%s.weight", name);
    params->weights = tensor_rand(weights_name, (int[]){output_dim, input_dim}, 2, limit);
    
    char bias_name[256];
    snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
    params->bias = tensor_rand(bias_name, (int[]){output_dim}, 1, limit);

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