#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn/parameters/batchnorm2d.h"

Parameters* batchnorm2d_parameters_create(const char *name, int num_features) 
{
    BatchNorm2DParameters *params = (BatchNorm2DParameters *)malloc(sizeof(BatchNorm2DParameters));

    char gamma_name[256];
    snprintf(gamma_name, sizeof(gamma_name), "%s.gamma", name);
    params->gamma = tensor_ones(gamma_name, (int[]){num_features}, 1);  // scale factor, initialized to 1

    char beta_name[256];
    snprintf(beta_name, sizeof(beta_name), "%s.beta", name);
    params->beta = tensor_zeros(beta_name, (int[]){num_features}, 1);  // shift factor, initialized to 0

    params->base.get_params = &batchnorm2d_get_params;
    params->base.num_params = 2;
    params->base.freeze_params = &batchnorm2d_parameters_freeze;
    params->base.free = &batchnorm2d_parameters_free;

    return (Parameters *)params;
}

Tensor **batchnorm2d_get_params(Parameters *base_params) 
{
    BatchNorm2DParameters *params = (BatchNorm2DParameters *)base_params;
    Tensor **param_array = (Tensor **)malloc(2 * sizeof(Tensor *));
    param_array[0] = params->gamma;
    param_array[1] = params->beta;
    return param_array;
}

void batchnorm2d_parameters_freeze(Parameters *base_params) 
{
    BatchNorm2DParameters *params = (BatchNorm2DParameters *)base_params;
    if (params->gamma) 
    {
        params->gamma->frozen = true;
    }
    if (params->beta) 
    {
        params->beta->frozen = true;
    }
}

void batchnorm2d_parameters_free(Parameters *base_params) 
{
    BatchNorm2DParameters *params = (BatchNorm2DParameters *)base_params;
    if (params->gamma) 
    {
        tensor_free(params->gamma);
    }
    if (params->beta) 
    {
        tensor_free(params->beta);
    }
    free(params);
}