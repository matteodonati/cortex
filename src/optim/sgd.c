#include <stdlib.h>
#include "optim/sgd.h"

Optimizer* sgd_create(float learning_rate) 
{
    SGD *optimizer = (SGD *)malloc(sizeof(SGD));
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.optim_type = OPTIM_TYPE_SGD;
    optimizer->base.step = &sgd_step;
    optimizer->base.free = &sgd_free;
    return (Optimizer *)optimizer;
}

void sgd_step(Optimizer *self, Tensor **params, int num_params) 
{
    for (int p = 0; p < num_params; p++) 
    {
        Tensor *param = params[p];
        for (int i = 0; i < param->size; i++) 
        {
            param->data[i] -= self->learning_rate * param->grad[i];
        }
    }
}

void sgd_free(Optimizer *self)
{
    free((SGD *)self);
}