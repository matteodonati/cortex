#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

#include "tensor/tensor.h"

typedef struct Parameters
{
    Tensor **(*get_params)(struct Parameters *self);
    void (*freeze_params)(struct Parameters *self);
    void (*free)(struct Parameters *self);
    int num_params;
} Parameters;

void parameters_free(Parameters *params);

#endif