#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

#include "tensor/tensor.h"

typedef struct Parameters
{
    int num_params;
    Tensor **(*get_params)(struct Parameters *self);
    void (*freeze_params)(struct Parameters *self);
    void (*free)(struct Parameters *self);
} Parameters;

void parameters_free(Parameters *params);

#endif