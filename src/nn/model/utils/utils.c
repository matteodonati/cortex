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

void model_summary(Model *model) 
{
    printf("=======================================================\n");
    printf("        Layer (type)        Output Shape        Param #\n");
    printf("-------------------------------------------------------\n");

    size_t total_params = 0;
    size_t trainable_params = 0;
    size_t non_trainable_params = 0;

    float input_size = 0.0f;
    float forward_size = 0.0f;
    float params_size = 0.0f;

    for (int i = 0; i < model->num_layers; i++) 
    {
        Layer *layer = model->layers[i];
        Parameters *params = layer->params;
        
        size_t num_layer_params = 0;
        size_t trainable_layer_params = 0;

        // Calculate the number of parameters and their sizes
        for (int j = 0; j < params->num_params; j++) 
        {
            Tensor *param = params->get_params(params)[j];
            size_t param_size = 1;
            for (int k = 0; k < param->ndim; k++) 
            {
                param_size *= param->shape[k];
            }
            num_layer_params += param_size;

            if (!param->frozen) 
            {
                trainable_layer_params += param_size;
            }

            // Accumulate parameter memory size
            params_size += param_size * sizeof(float) / (1024.0 * 1024.0);
        }

        // Output shape could be obtained from the layer's output tensor
        char output_shape[256] = "";
        sprintf(output_shape, "[-1");
        for (int dim = 0; dim < layer->output->ndim; dim++) 
        {
            sprintf(output_shape + strlen(output_shape), ", %d", layer->output->shape[dim]);
        }
        sprintf(output_shape + strlen(output_shape), "]");

        // Print the layer summary
        printf("%15s %24s %14zu\n", layer->name, output_shape, num_layer_params);

        total_params += num_layer_params;
        trainable_params += trainable_layer_params;
        non_trainable_params += (num_layer_params - trainable_layer_params);

        // Calculate memory sizes for input and forward/backward pass
        size_t output_size = 1;
        for (int dim = 0; dim < layer->output->ndim; dim++) 
        {
            output_size *= layer->output->shape[dim];
        }

        input_size += output_size * sizeof(float) / (1024.0 * 1024.0);
        forward_size += 2 * output_size * sizeof(float) / (1024.0 * 1024.0); // For both forward and backward pass
    }

    printf("-------------------------------------------------------\n");
    printf("Total parameters: %zu\n", total_params);
    printf("Trainable parameters: %zu\n", trainable_params);
    printf("Non-trainable parameters: %zu\n", non_trainable_params);
    printf("-------------------------------------------------------\n");
    printf("Input size (MB): %.2f\n", input_size);
    printf("Forward/backward pass size (MB): %.2f\n", forward_size);
    printf("Params size (MB): %.2f\n", params_size);
    printf("Estimated total size (MB): %.2f\n", input_size + forward_size + params_size);
    printf("=======================================================\n");
}