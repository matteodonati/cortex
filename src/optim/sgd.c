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
    for (int i = 0; i < num_params; i++) 
    {
        Tensor *param = params[i];
        if (param->frozen) 
        {
            continue;
        }

        for (int j = 0; j < param->size; j++) 
        {
            param->data[j] -= self->learning_rate * param->grad[j];
        }
    }
}

void sgd_free(Optimizer *self)
{
    free((SGD *)self);
}