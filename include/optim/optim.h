#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor/tensor.h"

#define OPTIM_TYPE_SGD 1

typedef struct Optimizer 
{
    float learning_rate;
    int optim_type;
    void (*step)(struct Optimizer *self, Tensor **params, int num_params);
    void (*free)(struct Optimizer *self);
} Optimizer;

void optimizer_free(Optimizer *optimizer);

#endif