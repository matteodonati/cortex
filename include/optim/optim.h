#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor/tensor.h"

#define OPTIM_TYPE_SGD 1
#define OPTIM_TYPE_ADAM 2
#define OPTIM_TYPE_ADAMW 3

typedef struct Optimizer 
{
    void (*step)(struct Optimizer *self, Tensor **params, int num_params);
    void (*free)(struct Optimizer *self);
    float learning_rate;
    int optim_type;
} Optimizer;

void optimizer_step(Optimizer *optim, Tensor **params, int num_params);
void optimizer_free(Optimizer *optimizer);

#endif