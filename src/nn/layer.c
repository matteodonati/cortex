#include <stdlib.h>
#include "nn/layer.h"
#include "tensor/tensor.h"

void layer_free(Layer *layer)
{
    if (layer)
    {
        if (layer->tensors)
        {
            for (int i = 0; i < layer->tensor_count; i++)
            {
                tensor_free(layer->tensors[i]);
            }
            free(layer->tensors);
        }
        if (layer->weights)
        {
            tensor_free(layer->weights);
        }
        if (layer->bias)
        {
            tensor_free(layer->bias);
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