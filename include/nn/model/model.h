#ifndef NN_MODEL_H
#define NN_MODEL_H

#include "nn/layers/layer.h"

typedef struct Model 
{
    Layer **layers;
    Tensor **params;
    int num_layers;
    int num_params;
} Model;

Model* model_create(Layer **layers, int num_layers);
void model_train(Model *self);
void model_eval(Model *self);
void model_zero_grad(Model *self);
void model_free(Model *self);

#endif