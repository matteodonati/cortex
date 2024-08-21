#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "optim/optim.h"

#define LAYER_TYPE_DENSE 1

typedef struct Layer 
{
    char *name;
    Tensor *weights;
    Tensor *bias;
    Tensor *input;
    Tensor *output;
    Tensor **tensors;
    int tensor_count;
    int layer_type;
    Tensor *(*forward)(struct Layer *self, Tensor *input);
    Tensor **(*get_params)(struct Layer *self, int *num_params);
    void (*freeze_params)(struct Layer *self);
    void (*free)(struct Layer *self);
} Layer;

void layer_free(Layer *layer);

#endif
