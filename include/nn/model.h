#ifndef NN_MODEL_H
#define NN_MODEL_H

#include "nn/layer.h"

typedef struct Model 
{
    Layer **layers;
    int num_layers;
    Tensor **params;
    int num_params;
} Model;

Model* model_create(Layer **layers, int num_layers);
void model_save(Model *model, const char *filename);
void model_load(Model *model, const char *filename);
void model_free(Model *model);

#endif