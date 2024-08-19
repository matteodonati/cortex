#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "tensor/tensor.h"
#include "optim/optim.h"

#define LAYER_TYPE_DENSE 1

typedef struct Layer 
{
    Tensor *weights;
    Tensor *bias;
    Tensor *input;
    Tensor *output;
    int layer_type;
    Tensor *(*forward)(struct Layer *self, Tensor *input);
    void (*update_params)(struct Layer *self, Optimizer *optimizer);
    void (*free)(struct Layer *self);
} Layer;

void layer_free(Layer *layer);

#endif
