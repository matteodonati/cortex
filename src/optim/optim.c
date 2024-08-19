#include <stdlib.h>
#include "tensor/tensor.h"
#include "ops/forward/forward.h"
#include "optim/optim.h"

Optimizer* create_sgd_optimizer(float learning_rate) 
{
    Optimizer *optimizer = (Optimizer *)malloc(sizeof(Optimizer));
    optimizer->learning_rate = learning_rate;
    optimizer->update = &sgd_update;
    return optimizer;
}

void sgd_update(Optimizer *self, Tensor *weights, Tensor *bias) 
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

void optimizer_free(Optimizer *optimizer) 
{
    if (optimizer) 
    {
        free(optimizer);
    }
}