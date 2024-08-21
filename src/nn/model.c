#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn/model.h"

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
        int num_params = 0;
        Tensor **layer_params = layers[i]->get_params(layers[i], &num_params);
        model->params = (Tensor **)realloc(model->params, (model->num_params + num_params) * sizeof(Tensor *));
        memcpy(model->params + model->num_params, layer_params, num_params * sizeof(Tensor *));
        model->num_params += num_params;
        free(layer_params);
    }

    return model;
}

void model_save(Model *model, const char *filename) 
{
    FILE *file = fopen(filename, "wb");
    if (!file) 
    {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }

    // Save the number of parameters
    fwrite(&model->num_params, sizeof(int), 1, file);

    // Save each tensor
    for (int i = 0; i < model->num_params; i++) 
    {
        Tensor *tensor = model->params[i];

        // Save the tensor name
        int name_len = strlen(tensor->name);
        fwrite(&name_len, sizeof(int), 1, file);
        fwrite(tensor->name, sizeof(char), name_len, file);

        // Save the tensor shape and data
        fwrite(&tensor->ndim, sizeof(int), 1, file);
        fwrite(tensor->shape, sizeof(int), tensor->ndim, file);
        fwrite(tensor->data, sizeof(float), tensor->size, file);
    }
    fclose(file);
}

void model_load(Model *model, const char *filename) 
{
    FILE *file = fopen(filename, "rb");
    if (!file) 
    {
        fprintf(stderr, "Error: Unable to open file %s for reading\n", filename);
        exit(EXIT_FAILURE);
    }

    int num_params;
    fread(&num_params, sizeof(int), 1, file);

    for (int i = 0; i < num_params; i++) 
    {
        // Read the tensor name
        int name_len;
        fread(&name_len, sizeof(int), 1, file);

        char *name = (char *)malloc((name_len + 1) * sizeof(char));
        fread(name, sizeof(char), name_len, file);
        name[name_len] = '\0';

        // Locate the corresponding tensor in the model
        Tensor *tensor = NULL;
        for (int j = 0; j < model->num_params; j++) 
        {
            if (strcmp(model->params[j]->name, name) == 0) 
            {
                tensor = model->params[j];
                break;
            }
        }

        if (tensor == NULL) 
        {
            fprintf(stderr, "Error: Tensor %s not found in the model\n", name);
            free(name);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Check number of dimensions
        int ndim;
        fread(&ndim, sizeof(int), 1, file);
        if (ndim != tensor->ndim) 
        {
            fprintf(stderr, "Error: Dimension mismatch for tensor %s\n", name);
            free(name);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Check the shape
        int *shape = (int *)malloc(ndim * sizeof(int));
        fread(shape, sizeof(int), ndim, file);
        for (int d = 0; d < ndim; d++) 
        {
            if (shape[d] != tensor->shape[d]) 
            {
                fprintf(stderr, "Error: Shape mismatch for tensor %s\n", name);
                free(shape);
                free(name);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
        
        // Read the tensor data
        fread(tensor->data, sizeof(float), tensor->size, file);

        free(shape);
        free(name);
    }
    fclose(file);
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