#ifndef SGD_H
#define SGD_H

#include "optim/optim.h"

typedef struct SGD 
{
    Optimizer base;
} SGD;

Optimizer* sgd_create(float learning_rate);
void sgd_step(Optimizer *self, Tensor **params, int num_params);
void sgd_free(Optimizer *self);

#endif