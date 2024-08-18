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

void sgd_update(Optimizer *self, Tensor *weights, Tensor *grad_weights, Tensor *bias, Tensor *grad_bias) 
{
    Tensor *delta_weights = tensor_scalar_mul(grad_weights, -self->learning_rate);
    Tensor *delta_bias = tensor_scalar_mul(grad_bias, -self->learning_rate);

    Tensor *new_weights = tensor_add(weights, delta_weights);
    Tensor *new_bias = tensor_add(bias, delta_bias);

    tensor_free(weights);
    tensor_free(bias);
    tensor_free(delta_weights);
    tensor_free(delta_bias);

    weights = new_weights;
    bias = new_bias;
}

void optimizer_free(Optimizer *optimizer) 
{
    if (optimizer) 
    {
        free(optimizer);
    }
}