#ifndef NN_MODEL_H
#define NN_MODEL_H

#include "nn/layer.h"
#include "tensor/tensor.h"

typedef struct Model 
{
    Layer **layers;
    int num_layers;
    Tensor **params;
    int num_params;
} Model;

Model* model_create(Layer **layers, int num_layers);
void model_free(Model *model);

#endif