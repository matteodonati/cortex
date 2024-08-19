#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor/tensor.h"

typedef struct Optimizer 
{
    void (*update)(struct Optimizer *self, Tensor *weights, Tensor *bias);
    float learning_rate;
    // Other parameters like momentum, beta1, beta2, etc.
} Optimizer;

Optimizer* create_sgd_optimizer(float learning_rate);
void sgd_update(Optimizer *self, Tensor *weights, Tensor *bias);
void optimizer_free(Optimizer *optimizer);

#endif