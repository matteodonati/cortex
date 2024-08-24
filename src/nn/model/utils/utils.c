#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nn/model/utils/utils.h"

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