#include <stdlib.h>
#include <string.h>
#include "utils/memory/pool.h"
#include "nn/layers/layer.h"

parameters_status_code_t parameters_destroy(parameters_t *params)
{
    if (params == NULL)
    {
        return PARAMETERS_DESTROY_FAILURE;
    }
    if (params->params_array)
    {
        if (pool_free(params->params_array) == POOL_FREE_FAILURE)
        {
            return PARAMETERS_DESTROY_FAILURE;
        }
    }
    if (params->free)
    {
        if (params->free(params) == PARAMETERS_DESTROY_FAILURE)
        {
            return PARAMETERS_DESTROY_FAILURE;
        }
    }
    else
    {
        if (pool_free(params) == POOL_FREE_FAILURE)
        {
            return PARAMETERS_DESTROY_FAILURE;
        }
    }
    return PARAMETERS_DESTROY_SUCCESS;
}

layer_status_code_t layer_destroy(layer_t *layer)
{
    if (layer == NULL)
    {
        return LAYER_DESTROY_FAILURE;
    }
    if (layer->name) 
    {
        if (pool_free(layer->name) == POOL_FREE_FAILURE)
        {
            return LAYER_DESTROY_FAILURE;
        }
    }
    if (layer->output)
    {
        if (tensor_destroy(layer->output) == TENSOR_DESTROY_FAILURE)
        {
            return LAYER_DESTROY_FAILURE;
        }
    }
    if (layer->params)
    {
        if (parameters_destroy(layer->params) == PARAMETERS_DESTROY_FAILURE)
        {
            return LAYER_DESTROY_FAILURE;
        }
    }
    if (layer->free)
    {
        if (layer->free(layer) == LAYER_DESTROY_FAILURE)
        {
            return LAYER_DESTROY_FAILURE;
        }
    }
    else
    {
        if (pool_free(layer) == POOL_FREE_FAILURE)
        {
            return LAYER_DESTROY_FAILURE;
        }
    }
    return LAYER_DESTROY_SUCCESS;
}