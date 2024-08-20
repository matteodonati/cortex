#ifndef SGD_H
#define SGD_H

#include "optim/optim.h"

typedef struct SGD 
{
    Optimizer base;
} SGD;

Optimizer* create_sgd_optimizer(float learning_rate);
void sgd_step(Optimizer *self, Tensor *weights, Tensor *bias);
void sgd_free(Optimizer *self);

#endif