#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"
#include "nn/parameters/dense.h"
#include "nn/layers/dense.h"

Layer* dense_create(const char *name, int input_dim, int output_dim) 
{
    Dense *dense = (Dense *)malloc(sizeof(Dense));

    dense->input_dim = input_dim;
    dense->output_dim = output_dim;
    
    dense->base.name = NULL;
    if (name) 
    {
        dense->base.name = (char *)malloc((strlen(name) + 1) * sizeof(char));
        strcpy(dense->base.name, name);
    }
    dense->base.layer_type = LAYER_TYPE_DENSE;
    dense->base.params = dense_parameters_create(name, input_dim, output_dim);
    dense->base.forward = &dense_forward;
    dense->base.free = &dense_free;

    return (Layer *)dense;
}

Tensor* dense_forward(Layer *self, Tensor *input) 
{
    // Get trainable params
    DenseParameters *params = (DenseParameters *)self->params;

    // Store the input tensor
    self->input = input;

    // Forward pass
    Tensor *transposed_weights = tensor_transpose(params->weights, (int[]){1, 0});
    Tensor *z = tensor_matmul(input, transposed_weights);
    Tensor *output = tensor_add(z, params->bias);

    // Store the created tensors in the tensors list
    self->tensors = (Tensor **)malloc(3 * sizeof(Tensor *));
    self->tensor_count = 0;
    self->tensors[self->tensor_count++] = transposed_weights;
    self->tensors[self->tensor_count++] = z;
    self->tensors[self->tensor_count++] = output;

    // Store the output tensor
    self->output = output;

    return output;
}

void dense_free(Layer *self) 
{
    free((Dense *)self);
}