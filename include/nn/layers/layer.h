#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "tensor/tensor.h"

typedef struct parameters
{
    size_t num_params;
    tensor_t **params_array;
    void (*freeze_params)(struct parameters *self);
    parameters_status_code_t (*free)(struct parameters *self);
} parameters_t;

typedef struct layer_t
{
    char *name;
    bool is_training;
    tensor_t *input;
    tensor_t *output;
    parameters_t *params;
    tensor_t *(*forward)(struct layer_t *self, const tensor_t *input);
    layer_status_code_t (*free)(struct layer_t *self);
} layer_t;

parameters_status_code_t parameters_destroy(parameters_t *params);
layer_status_code_t layer_destroy(layer_t *layer);

static inline tensor_t *layer_forward(layer_t *self, const tensor_t *x)
{
    return self->forward(self, x);
}

#endif