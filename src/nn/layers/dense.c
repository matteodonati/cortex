#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "utils/memory/pool.h"
#include "nn/layers/dense.h"

parameters_t* dense_parameters_create(size_t input_dim, size_t output_dim)
{
    dense_parameters_t *params = (dense_parameters_t *)pool_alloc(sizeof(dense_parameters_t));
    if (params == NULL)
    {
        return NULL;
    }

    params->base.freeze_params = dense_parameters_freeze;
    params->base.free = dense_parameters_destroy;
    params->base.num_params = 2;

    float limit = sqrtf(1.0f / input_dim);

    size_t weights_shape[2] = {output_dim, input_dim};
    params->weights = tensor_rand(weights_shape, 2, limit);
    if (params->weights == NULL)
    {
        pool_free(params);
        return NULL;
    }

    size_t bias_shape[1] = {output_dim};
    params->bias = tensor_rand(bias_shape, 1, limit);
    if (params->bias == NULL)
    {
        tensor_destroy(params->weights);
        pool_free(params);
        return NULL;
    }

    params->base.params_array = (tensor_t **)pool_alloc(2 * sizeof(tensor_t *));
    if (params->base.params_array == NULL)
    {
        tensor_destroy(params->bias);
        tensor_destroy(params->weights);
        pool_free(params);
        return NULL;
    }
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

parameters_status_code_t dense_parameters_destroy(parameters_t *self)
{
    if (self == NULL)
    {
        return PARAMETERS_DESTROY_FAILURE;
    }

    dense_parameters_t *params = (dense_parameters_t *)self;

    if (params->weights)
    {
        if (tensor_destroy(params->weights) == TENSOR_DESTROY_FAILURE)
        {
            return PARAMETERS_DESTROY_FAILURE;
        }
    }
    if (params->bias)
    {
        if (tensor_destroy(params->bias) == TENSOR_DESTROY_FAILURE)
        {
            return PARAMETERS_DESTROY_FAILURE;
        }
    }
    if (pool_free(params) == POOL_FREE_FAILURE)
    {
        return PARAMETERS_DESTROY_FAILURE;
    }

    return PARAMETERS_DESTROY_SUCCESS;
}

layer_t* dense_create(const char *name, size_t input_dim, size_t output_dim)
{
    dense_layer_t *dense = (dense_layer_t *)pool_alloc(sizeof(dense_layer_t));
    if (dense == NULL)
    {
        return NULL;
    }

    dense->input_dim = input_dim;
    dense->output_dim = output_dim;

    dense->base.name = NULL;
    if (name)
    {
        size_t name_length = strlen(name) + 1;
        dense->base.name = (char *)pool_alloc(name_length * sizeof(char));
        if (dense->base.name == NULL)
        {
            pool_free(dense);
            return NULL;
        }
        memcpy(dense->base.name, name, name_length);
    }
    dense->base.is_training = false;
    dense->base.forward = dense_forward;
    dense->base.free = dense_destroy;

    dense->base.params = dense_parameters_create(input_dim, output_dim);
    if (dense->base.params == NULL)
    {
        if (dense->base.name)
        {
            pool_free(dense->base.name);
        }
        pool_free(dense);
        return NULL;
    }

    return (layer_t *)dense;
}

tensor_t* dense_forward(layer_t *self, const tensor_t *input)
{
    if (self == NULL || input == NULL)
    {
        return NULL;
    }

    dense_layer_t *dense = (dense_layer_t *)self;
    dense_parameters_t *params = (dense_parameters_t *)self->params;

    if (input->ndim != 2)
    {
        return NULL;
    }
    if (input->shape[1] != dense->input_dim)
    {
        return NULL;
    }

    size_t batch_size = input->shape[0];
    size_t output_dim = dense->output_dim;
    size_t input_dim = dense->input_dim;

    size_t output_shape[2] = {batch_size, output_dim};
    tensor_t *output = tensor_zeros(output_shape, 2);
    if (output == NULL)
    {
        return NULL;
    }

    const float *input_data = input->data;
    const float *weights_data = params->weights->data;
    const float *bias_data = params->bias->data;
    float *output_data = output->data;

    for (size_t i = 0; i < batch_size; ++i)
    {
        for (size_t j = 0; j < output_dim; ++j)
        {
            float sum = bias_data[j];
            const float *input_row = &input_data[i * input_dim];
            const float *weight_row = &weights_data[j * input_dim];

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
    if (output == NULL || output->grad == NULL)
    {
        return;
    }

    layer_t *layer = (layer_t *)output->context;
    if (layer == NULL)
    {
        return;
    }

    dense_layer_t *dense = (dense_layer_t *)layer;
    dense_parameters_t *params = (dense_parameters_t *)layer->params;
    tensor_t *input = layer->input;

    if (input == NULL || input->grad == NULL)
    {
        return;
    }

    size_t batch_size = input->shape[0];
    size_t output_dim = dense->output_dim;
    size_t input_dim = dense->input_dim;

    const float *output_grad = output->grad;
    const float *input_data = input->data;
    float *input_grad = input->grad;
    const float *weights_data = params->weights->data;
    float *weights_grad = params->weights->grad;
    float *bias_grad = params->bias->grad;

    for (size_t i = 0; i < batch_size; ++i)
    {
        const float *input_row = &input_data[i * input_dim];
        float *input_grad_row = &input_grad[i * input_dim];

        for (size_t j = 0; j < output_dim; ++j)
        {
            float grad_out = output_grad[i * output_dim + j];

            bias_grad[j] += grad_out;

            float *weights_grad_row = &weights_grad[j * input_dim];
            const float *weights_row = &weights_data[j * input_dim];

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

layer_status_code_t dense_destroy(layer_t *self)
{
    if (self == NULL)
    {
        return LAYER_DESTROY_FAILURE;
    }

    dense_layer_t *dense = (dense_layer_t *)self;
    
    if (pool_free(dense) == POOL_FREE_FAILURE)
    {
        return LAYER_DESTROY_FAILURE;
    }
    
    return LAYER_DESTROY_SUCCESS;
}