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
    dense->base.is_training = false;
    dense->base.forward = &dense_forward;
    dense->base.free = &dense_free;

    return (Layer *)dense;
}

Tensor* dense_forward(Layer *self, Tensor *x) 
{
    // Get trainable params
    DenseParameters *params = (DenseParameters *)self->params;

    if (x->ndim != 2) 
    {
        fprintf(stderr, "Error: Input tensor must be 2-dimensional in dense_forward.\n");
        exit(EXIT_FAILURE);
    }
    
    if (x->shape[1] != params->weights->shape[1]) 
    {
        fprintf(stderr, "Error: Input tensor has incompatible shape in dense_forward.\n");
        fprintf(stderr, "Expected %d columns, but got %d columns.\n", params->weights->shape[1], x->shape[1]);
        exit(EXIT_FAILURE);
    }

    // Forward pass
    Tensor *weights_T = tensor_transpose(params->weights, (int[]){1, 0});
    Tensor *z = tensor_matmul(x, weights_T);
    Tensor *y = tensor_add(z, params->bias);

    // Pointers to intermediate results
    self->input = x;
    self->tensor_count = 3;
    self->tensors = (Tensor **)malloc(self->tensor_count * sizeof(Tensor *));
    self->tensors[0] = weights_T;
    self->tensors[1] = z;
    self->tensors[2] = y;
    self->output = y;

    return y;
}

void dense_free(Layer *self) 
{
    if (self)
    {
        free((Dense *)self);
    }
}