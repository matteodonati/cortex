#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "utils/memory/pool.h"
#include "nn/layers/dense.h"

parameters_t* dense_parameters_create(size_t input_dim, size_t output_dim)
{
    dense_parameters_t *params = (dense_parameters_t *)pool_alloc(sizeof(dense_parameters_t));
    assert(params != NULL && "Failed to allocate memory for dense_parameters_t.");

    params->base.freeze_params = dense_parameters_freeze;
    params->base.free = dense_parameters_free;
    params->base.num_params = 2;

    float limit = sqrtf(1.0f / input_dim);

    size_t weights_shape[2] = {output_dim, input_dim};
    params->weights = tensor_rand(weights_shape, 2, limit);
    assert(params->weights != NULL && "Failed to create weights tensor.");

    size_t bias_shape[1] = {output_dim};
    params->bias = tensor_rand(bias_shape, 1, limit);
    assert(params->bias != NULL && "Failed to create bias tensor.");

    params->base.params_array = (tensor_t **)pool_alloc(2 * sizeof(tensor_t *));
    assert(params->base.params_array != NULL && "Failed to allocate params_array.");
    params->base.params_array[0] = params->weights;
    params->base.params_array[1] = params->bias;

    return (parameters_t *)params;
}

void dense_parameters_freeze(parameters_t *self)
{
    dense_parameters_t *params = (dense_parameters_t *)self;
    params->weights->frozen = true;
    params->bias->frozen = true;
}

void dense_parameters_free(parameters_t *self)
{
    if (self == NULL)
    {
        return;
    }

    dense_parameters_t *params = (dense_parameters_t *)self;

    if (params->weights)
    {
        tensor_destroy(params->weights);
    }
    if (params->bias)
    {
        tensor_destroy(params->bias);
    }
    if (params->base.params_array)
    {
        pool_free(params->base.params_array);
    }
    pool_free(params);
}

layer_t* dense_create(const char *name, size_t input_dim, size_t output_dim)
{
    dense_layer_t *dense = (dense_layer_t *)pool_alloc(sizeof(dense_layer_t));
    assert(dense != NULL && "Failed to allocate memory for dense_layer_t.");

    dense->input_dim = input_dim;
    dense->output_dim = output_dim;

    dense->base.name = NULL;
    if (name)
    {
        size_t name_length = strlen(name) + 1;
        dense->base.name = (char *)pool_alloc(name_length * sizeof(char));
        assert(dense->base.name != NULL && "Failed to allocate memory for layer name.");
        memcpy(dense->base.name, name, name_length);
    }
    dense->base.is_training = false;
    dense->base.forward = dense_forward;
    dense->base.free = dense_free;

    dense->base.params = dense_parameters_create(input_dim, output_dim);
    assert(dense->base.params != NULL && "Failed to create dense parameters.");

    return (layer_t *)dense;
}

tensor_t* dense_forward(layer_t *self, const tensor_t *input)
{
    assert(self != NULL && "Layer pointer is NULL in dense_forward.");
    assert(input != NULL && "Input tensor is NULL in dense_forward.");

    dense_layer_t *dense = (dense_layer_t *)self;
    dense_parameters_t *params = (dense_parameters_t *)self->params;

    assert(input->ndim == 2 && "Input tensor must be 2-dimensional in dense_forward.");
    assert(input->shape[1] == dense->input_dim && "Input tensor has incompatible shape in dense_forward.");

    size_t batch_size = input->shape[0];
    size_t output_dim = dense->output_dim;
    size_t input_dim = dense->input_dim;

    size_t output_shape[2] = {batch_size, output_dim};
    tensor_t *output = tensor_zeros(output_shape, 2);
    assert(output != NULL && "Failed to create output tensor in dense_forward.");

    const float *__restrict__ input_data = input->data;
    const float *__restrict__ weights_data = params->weights->data;
    const float *__restrict__ bias_data = params->bias->data;
    float *__restrict__ output_data = output->data;

    for (size_t i = 0; i < batch_size; ++i)
    {
        for (size_t j = 0; j < output_dim; ++j)
        {
            float sum = bias_data[j];
            const float *__restrict__ input_row = &input_data[i * input_dim];
            const float *__restrict__ weight_row = &weights_data[j * input_dim];

            for (size_t k = 0; k < input_dim; ++k)
            {
                sum += input_row[k] * weight_row[k];
            }
            output_data[i * output_dim + j] = sum;
        }
    }

    self->output = output;
    self->input = (tensor_t *)input;

    output->backward = dense_backward;
    output->context = self;

    return output;
}

void dense_backward(tensor_t *output)
{
    assert(output != NULL && "Output tensor is NULL in dense_backward.");
    assert(output->grad != NULL && "Output tensor gradient is NULL in dense_backward.");

    layer_t *layer = (layer_t *)output->context;
    assert(layer != NULL && "Layer is NULL in dense_backward.");

    dense_layer_t *dense = (dense_layer_t *)layer;
    dense_parameters_t *params = (dense_parameters_t *)layer->params;
    tensor_t *input = layer->input;

    size_t batch_size = input->shape[0];
    size_t output_dim = dense->output_dim;
    size_t input_dim = dense->input_dim;

    const float *__restrict__ output_grad = output->grad;
    const float *__restrict__ input_data = input->data;
    float *__restrict__ input_grad = input->grad;
    const float *__restrict__ weights_data = params->weights->data;
    float *__restrict__ weights_grad = params->weights->grad;
    float *__restrict__ bias_grad = params->bias->grad;

    for (size_t i = 0; i < batch_size; ++i)
    {
        const float *__restrict__ input_row = &input_data[i * input_dim];
        float *__restrict__ input_grad_row = &input_grad[i * input_dim];

        for (size_t j = 0; j < output_dim; ++j)
        {
            float grad_out = output_grad[i * output_dim + j];

            bias_grad[j] += grad_out;

            float *__restrict__ weights_grad_row = &weights_grad[j * input_dim];
            const float *__restrict__ weights_row = &weights_data[j * input_dim];

            for (size_t k = 0; k < input_dim; ++k)
            {
                weights_grad_row[k] += grad_out * input_row[k];
                input_grad_row[k] += grad_out * weights_row[k];
            }
        }
    }

    if (input->backward)
    {
        input->backward(input);
    }
}

void dense_free(layer_t *self)
{
    if (self == NULL)
    {
        return;
    }

    dense_layer_t *dense = (dense_layer_t *)self;
    
    pool_free(dense);
}