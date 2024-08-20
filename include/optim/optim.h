#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor/tensor.h"

#define OPTIM_TYPE_SGD 1

typedef struct Optimizer 
{
    float learning_rate;
    int optim_type;
    void (*step)(struct Optimizer *self, Tensor *weights, Tensor *bias);
    void (*free)(struct Optimizer *self);
} Optimizer;

void optimizer_free(Optimizer *optimizer);

#endif