#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "nn/parameters/parameters.h"
#include "optim/optim.h"

#define LAYER_TYPE_DENSE 1
#define LAYER_TYPE_CONV2D 2
#define LAYER_TYPE_MAXPOOL 3

typedef struct Layer 
{
    char *name;
    Parameters *params;
    Tensor *input;
    Tensor *output;
    Tensor **tensors;
    int tensor_count;
    int layer_type;
    Tensor *(*forward)(struct Layer *self, Tensor *input);
    void (*free)(struct Layer *self);
} Layer;

Tensor *layer_forward(Layer *self, Tensor *x);
void layer_free(Layer *layer);

#endif
