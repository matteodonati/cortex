#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn/dense.h"
#include "tensor/tensor.h"
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"
#include "optim/optim.h"

Layer* dense_create(const char *name, int input_dim, int output_dim)
{
    Dense *dense = (Dense *)malloc(sizeof(Dense));
    
    dense->base.name = (char *)malloc((strlen(name) + 1) * sizeof(char));
    strcpy(dense->base.name, name);
    
    dense->input_dim = input_dim;
    dense->output_dim = output_dim;
    dense->base.layer_type = LAYER_TYPE_DENSE;

    char weights_name[256];
    snprintf(weights_name, sizeof(weights_name), "%s.weight", name);
    dense->base.weights = tensor_rand(weights_name, (int[]){output_dim, input_dim}, 2);

    char bias_name[256];
    snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
    dense->base.bias = tensor_zeros(bias_name, (int[]){output_dim}, 1);

    dense->base.forward = &dense_forward;
    dense->base.get_params = &dense_get_params;
    dense->base.freeze_params = &dense_freeze_params;
    dense->base.free = &dense_free;
    
    return (Layer *)dense;
}

Tensor* dense_forward(Layer *self, Tensor *input)
{
    // Store the input tensor
    self->input = input;

    // Forward pass
    Tensor *transposed_weights = tensor_transpose(self->weights, (int[]){1, 0});
    Tensor *z = tensor_matmul(input, transposed_weights);
    Tensor *output = tensor_add(z, self->bias);

    // Store the created tensors in the tensors list
    self->tensors = (Tensor **)malloc(3 * sizeof(Tensor *));
    self->tensor_count = 0;
    self->tensors[self->tensor_count++] = transposed_weights;
    self->tensors[self->tensor_count++] = z;
    self->tensors[self->tensor_count++] = output;

    // Set up backward functions and references
    output->grad_a = z;
    output->grad_b = self->bias;
    output->backward = tensor_add_backward;
    z->grad_a = input;
    z->grad_b = transposed_weights;
    z->backward = tensor_matmul_backward;

    // Store the output tensor
    self->output = output;

    return output;
}

Tensor** dense_get_params(Layer *self, int *num_params) 
{
    Tensor **params = (Tensor **)malloc(2 * sizeof(Tensor *));
    params[0] = self->weights;
    params[1] = self->bias;
    *num_params = 2;
    return params;
}

void dense_freeze_params(Layer *self) 
{
    if (self->weights) 
    {
        self->weights->frozen = 1;
    }
    if (self->bias) 
    {
        self->bias->frozen = 1;
    }
}

void dense_free(Layer *self)
{
    free((Dense *)self);
}