#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "utils/memory/pool.h"
#include "nn/layers/layer.h"

void layer_free(layer_t *layer)
{
    if (layer == NULL)
    {
        return;
    }

    if (layer->name) 
    {
        pool_free(layer->name);
    }
    if (layer->output)
    {
        tensor_destroy(layer->output);
    }
    if (layer->params)
    {
        if (layer->params->free)
        {
            layer->params->free(layer->params);
        }
        else
        {
            pool_free(layer->params);
        }
    }
    if (layer->free)
    {
        layer->free(layer);
    }
    else
    {
        pool_free(layer);
    }
}