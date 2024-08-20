#include <stdlib.h>
#include "optim/sgd.h"

Optimizer* create_sgd_optimizer(float learning_rate) 
{
    SGD *optimizer = (SGD *)malloc(sizeof(SGD));
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.optim_type = OPTIM_TYPE_SGD;
    optimizer->base.step = &sgd_step;
    optimizer->base.free = &sgd_free;
    return (Optimizer *)optimizer;
}

void sgd_step(Optimizer *self, Tensor *weights, Tensor *bias) 
{
    for (int i = 0; i < weights->size; i++) 
    {
        weights->data[i] -= self->learning_rate * weights->grad[i];
    }
    for (int i = 0; i < bias->size; i++) 
    {
        bias->data[i] -= self->learning_rate * bias->grad[i];
    }
}

void sgd_free(Optimizer *self)
{
    free((SGD *)self);
}