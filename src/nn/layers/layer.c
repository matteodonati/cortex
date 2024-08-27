#include <stdlib.h>
#include "nn/layers/layer.h"

void layer_free(Layer *layer)
{
    if (layer)
    {
        if (layer->params)
        {
            parameters_free(layer->params);
        }
        if (layer->name) 
        {
            free(layer->name);
        }
        if (layer->free)
        {
            layer->free(layer);
        }
        else
        {
            free(layer);
        }
    }
}