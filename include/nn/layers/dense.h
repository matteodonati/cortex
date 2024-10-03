#ifndef NN_DENSE_H
#define NN_DENSE_H

#include "nn/layers/layer.h"

typedef struct dense_parameters
{
    parameters_t base;
    tensor_t *weights;
    tensor_t *bias;
} dense_parameters_t;

typedef struct dense_layer_t
{
    layer_t base;
    size_t input_dim;
    size_t output_dim;
} dense_layer_t;

parameters_t* dense_parameters_create(size_t input_dim, size_t output_dim);
void dense_parameters_freeze(parameters_t *self);
parameters_status_code_t dense_parameters_destroy(parameters_t *self);

layer_t* dense_create(const char *name, size_t input_dim, size_t output_dim);
tensor_t* dense_forward(layer_t *self, const tensor_t *input);
void dense_backward(tensor_t *output);
layer_status_code_t dense_destroy(layer_t *self);

#endif