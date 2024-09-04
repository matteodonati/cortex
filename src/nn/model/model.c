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

void model_train(Model *self)
{
    for (int i = 0; i < self->num_layers; i++)
    {
        self->layers[i]->is_training = true;
    }
}

void model_eval(Model *self)
{
    for (int i = 0; i < self->num_layers; i++)
    {
        self->layers[i]->is_training = false;
    }
}

void model_zero_grad(Model *self)
{
    if (self == NULL || self->params == NULL) 
    {
        return;
    }

    for (int i = 0; i < self->num_params; i++) 
    {
        Tensor *param = self->params[i];
        if (param->grad) 
        {
            memset(param->grad, 0, param->size * sizeof(float));
        }
    }
}

void model_free(Model *self)
{
    if (self) 
    {
        for (int i = 0; i < self->num_layers; i++) 
        {
            layer_free(self->layers[i]);
        }
        free(self->params);
        free(self);
    }
}