#ifndef NN_DENSE_PARAMETERS_H
#define NN_DENSE_PARAMETERS_H

#include "nn/parameters/parameters.h"

typedef struct DenseParameters
{
    Parameters base;
    Tensor *weights;
    Tensor *bias;
} DenseParameters;

Parameters* dense_parameters_create(const char *name, int input_dim, int output_dim);
Tensor **dense_get_params(Parameters *base_params);
void dense_parameters_freeze(Parameters *base_params);
void dense_parameters_free(Parameters *base_params);

#endif