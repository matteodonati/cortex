#include <stdlib.h>
#include <string.h>
#include "nn/model/model.h"

Model* model_create(Layer **layers, int num_layers) 
{
    Model *model = (Model *)malloc(sizeof(Model));
    model->layers = layers;
    model->num_layers = num_layers;
    model->params = NULL;
    model->num_params = 0;

    // Collect all parameters from the layers
    for (int i = 0; i < num_layers; i++) 
    {
        Parameters *layer_params = layers[i]->params;
        if (layer_params && layer_params->get_params) 
        {
            Tensor **params = layer_params->get_params(layer_params);
            int num_params = layer_params->num_params;

            model->params = (Tensor **)realloc(model->params, (model->num_params + num_params) * sizeof(Tensor *));
            memcpy(model->params + model->num_params, params, num_params * sizeof(Tensor *));
            model->num_params += num_params;
            
            free(params);
        }
    }

    return model;
}

void model_zero_grad(Model *model)
{
    if (model == NULL || model->params == NULL) 
    {
        return;
    }

    for (int i = 0; i < model->num_params; i++) 
    {
        Tensor *param = model->params[i];
        if (param->grad) 
        {
            memset(param->grad, 0, param->size * sizeof(float));
        }
    }
}

void model_free(Model *model)
{
    if (model) 
    {
        for (int i = 0; i < model->num_layers; i++) 
        {
            layer_free(model->layers[i]);
        }
        // Free model->layers only in case Layer **layers is created with malloc (i.e., it is not on the stack).
        // free(model->layers);
        free(model->params);
        free(model);
    }
}