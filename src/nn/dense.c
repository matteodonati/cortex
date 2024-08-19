#include <stdlib.h>
#include <string.h>
#include "nn/dense.h"
#include "tensor/tensor.h"
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"
#include "optim/optim.h"

#include <stdio.h>
#include "tensor/utils/utils.h"

Layer* dense_create(int input_dim, int output_dim)
{
    Dense *dense = (Dense *)malloc(sizeof(Dense));
    dense->input_dim = input_dim;
    dense->output_dim = output_dim;
    dense->base.layer_type = LAYER_TYPE_DENSE;
    dense->base.weights = tensor_rand((int[]){output_dim, input_dim}, 2);
    dense->base.bias = tensor_zeros((int[]){output_dim}, 1);
    dense->base.forward = &dense_forward;
    dense->base.backward = &dense_backward;
    dense->base.update_params = &dense_update_params;
    dense->base.free = &dense_free;
    return (Layer *)dense;
}

Tensor* dense_forward(Layer *self, Tensor *input)
{
    self->input = input;
    Tensor *z = tensor_matmul(input, tensor_transpose(self->weights, (int[]){1, 0}));
    Tensor *output = tensor_add(z, self->bias);
    tensor_free(z);
    self->output = output;
    return output;
}

void dense_backward(Layer *self, float *grad)
{
    Tensor *grad_output = tensor_from_array(grad, self->output->shape, self->output->ndim);

    // Calculate gradients
    Tensor *grad_weights = tensor_matmul(tensor_transpose(grad_output, (int[]){1, 0}), self->input);
    Tensor *grad_bias = tensor_sum(grad_output, 0);
    Tensor *grad_input = tensor_matmul(grad_output, self->weights);

    // Store gradients in the grad field of the respective Tensors
    memcpy(self->weights->grad, grad_weights->data, grad_weights->size * sizeof(float));
    memcpy(self->bias->grad, grad_bias->data, grad_bias->size * sizeof(float));
    memcpy(self->input->grad, grad_input->data, grad_input->size * sizeof(float));

    // Free the temporary gradient tensors
    tensor_free(grad_weights);
    tensor_free(grad_bias);
    tensor_free(grad_input);
    tensor_free(grad_output);
}

void dense_update_params(Layer *self, Optimizer *optimizer)
{
    optimizer->update(optimizer, self->weights, self->bias);
}

void dense_free(Layer *self)
{
    free(self);
}