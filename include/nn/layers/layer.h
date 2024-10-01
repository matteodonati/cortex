#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "tensor/tensor.h" 

typedef struct parameters
{
    size_t num_params;
    tensor_t **params_array;
    void (*freeze_params)(struct parameters *self);
    void (*free)(struct parameters *self);
} parameters_t;

typedef struct layer_t
{
    char *name;
    bool is_training;
    tensor_t *input;
    tensor_t *output;
    parameters_t *params;
    tensor_t *(*forward)(struct layer_t *self, const tensor_t *input);
    void (*free)(struct layer_t *self);
} layer_t;

static inline tensor_t *layer_forward(layer_t *self, const tensor_t *x)
{
    return self->forward(self, x);
}

void layer_free(layer_t *layer);

#endif