#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"
#include "nn/parameters/batchnorm2d.h"
#include "nn/layers/batchnorm2d.h"

Layer* batchnorm2d_create(const char *name, int num_features, float epsilon, float momentum) 
{
    BatchNorm2D *bn = (BatchNorm2D *)malloc(sizeof(BatchNorm2D));

    bn->num_features = num_features;
    bn->epsilon = epsilon;
    bn->momentum = momentum;
    bn->running_mean = tensor_zeros(NULL, (int[]){num_features}, 1);
    bn->running_var = tensor_ones(NULL, (int[]){num_features}, 1);
    bn->base.name = NULL;
    if (name) 
    {
        bn->base.name = (char *)malloc((strlen(name) + 1) * sizeof(char));
        strcpy(bn->base.name, name);
    }
    bn->base.layer_type = LAYER_TYPE_BATCHNORM2D;
    bn->base.params = batchnorm2d_parameters_create(name, num_features);
    bn->base.forward = &batchnorm2d_forward;
    bn->base.free = &batchnorm2d_free;

    return (Layer *)bn;
}

Tensor* batchnorm2d_forward(Layer *self, Tensor *x) 
{
    BatchNorm2D *bn = (BatchNorm2D *)self;
    BatchNorm2DParameters *params = (BatchNorm2DParameters *)self->params;

    // Input information
    int batch_size = x->shape[0];
    int num_features = x->shape[1];
    int height = x->shape[2];
    int width = x->shape[3];
    int num_elements = batch_size * height * width;

    // Tensors
    Tensor *mean = NULL;
    Tensor *var = NULL;

    if (self->is_training) 
    {
        // Use computed mean and variance during training
        mean = tensor_zeros(NULL, (int[]){num_features}, 1);
        var = tensor_zeros(NULL, (int[]){num_features}, 1);

        // Update running mean and variance using momentum
        for (int c = 0; c < num_features; c++) 
        {
            bn->running_mean->data[c] = (1 - bn->momentum) * bn->running_mean->data[c] + bn->momentum * mean->data[c];
            float unbiased_var = var->data[c] * (float)num_elements / (num_elements - 1);  // unbiased variance
            bn->running_var->data[c] = (1 - bn->momentum) * bn->running_var->data[c] + bn->momentum * unbiased_var;
        }
    } 
    else 
    {
        // Use running mean and variance during evaluation
        mean = tensor_clone(NULL, bn->running_mean);
        var = tensor_clone(NULL, bn->running_var);
    }

    // Normalize
    Tensor *normalized_x = tensor_normalize2d(x, self->is_training, mean, var, (int []){0, 2, 3}, 3, bn->epsilon);

    // Scale and shift
    Tensor *gamma_reshaped = tensor_reshape(params->gamma, (int[]){num_features, 1, 1}, 3);
    Tensor *beta_reshaped = tensor_reshape(params->beta, (int[]){num_features, 1, 1}, 3);
    Tensor *scaled_x = tensor_mul(normalized_x, gamma_reshaped);
    Tensor *y = tensor_add(scaled_x, beta_reshaped);

    // Store intermediate tensors
    self->input = x;
    self->output = y;
    self->tensor_count = 7;
    self->tensors = (Tensor **)malloc(self->tensor_count * sizeof(Tensor *));
    self->tensors[0] = mean;
    self->tensors[1] = var;
    self->tensors[2] = normalized_x;
    self->tensors[3] = gamma_reshaped;
    self->tensors[4] = beta_reshaped;
    self->tensors[5] = scaled_x;
    self->tensors[6] = y;

    return y;
}

void batchnorm2d_free(Layer *self) 
{
    if (self)
    {
        BatchNorm2D *bn = (BatchNorm2D *)self;
        tensor_free(bn->running_mean);
        tensor_free(bn->running_var);
        free(bn);
    }
}