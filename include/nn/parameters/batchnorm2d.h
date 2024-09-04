#ifndef NN_BATCHNORM2D_PARAMETERS_H
#define NN_BATCHNORM2D_PARAMETERS_H

#include "nn/parameters/parameters.h"

typedef struct BatchNorm2DParameters
{
    Parameters base;
    Tensor *gamma;
    Tensor *beta;
} BatchNorm2DParameters;

Parameters* batchnorm2d_parameters_create(const char *name, int num_features);
Tensor **batchnorm2d_get_params(Parameters *base_params);
void batchnorm2d_parameters_freeze(Parameters *base_params);
void batchnorm2d_parameters_free(Parameters *base_params);

#endif