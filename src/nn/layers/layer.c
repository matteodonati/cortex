#include <stdlib.h>
#include "nn/layers/layer.h"

Tensor *layer_forward(Layer *self, Tensor *x)
{
    if (self->forward)
    {
        return self->forward(self, x);
    }
    return NULL;
}

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