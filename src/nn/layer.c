#include <stdlib.h>
#include "nn/layer.h"
#include "tensor/tensor.h"

void layer_free(Layer *layer)
{
    if (layer->weights)
    {
        tensor_free(layer->weights);
    }
    if (layer->bias)
    {
        tensor_free(layer->bias);
    }
    if (layer->grad_weights)
    {
        tensor_free(layer->grad_weights);
    }
    if (layer->grad_bias)
    {
        tensor_free(layer->grad_bias);
    }
    /*
    if (layer->output)
    {
        tensor_free(layer->output);
    }
    if (layer->input)
    {
        tensor_free(layer->input);
    }
    */
    if (layer->free)
    {
        layer->free(layer);
    }
    else
    {
        free(layer);
    }
}
