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

    // Total number of elements across batch, height, and width
    int num_elements = batch_size * height * width;

    // Batch norm tensors. Shape is {num_features}
    Tensor *mean = tensor_zeros(NULL, (int[]){num_features}, 1);
    Tensor *var = tensor_zeros(NULL, (int[]){num_features}, 1);

    if (self->is_training) 
    {
        // Compute mean over batch, height, width
        for (int n = 0; n < batch_size; n++) 
        {
            for (int c = 0; c < num_features; c++) 
            {
                for (int h = 0; h < height; h++) 
                {
                    for (int w = 0; w < width; w++) 
                    {
                        mean->data[c] += x->data[((n * num_features + c) * height + h) * width + w];
                    }
                }
            }
        }
        for (int c = 0; c < num_features; c++) 
        {
            mean->data[c] /= num_elements;
        }

        // Compute biased variance over batch, height, width
        for (int n = 0; n < batch_size; n++) 
        {
            for (int c = 0; c < num_features; c++) 
            {
                for (int h = 0; h < height; h++) 
                {
                    for (int w = 0; w < width; w++) 
                    {
                        float diff = x->data[((n * num_features + c) * height + h) * width + w] - mean->data[c];
                        var->data[c] += diff * diff;
                    }
                }
            }
        }
        for (int c = 0; c < num_features; c++) 
        {
            var->data[c] /= num_elements;
        }

        // Update running mean and running variance
        for (int c = 0; c < num_features; c++) 
        {
            // Update running mean
            bn->running_mean->data[c] = (1 - bn->momentum) * bn->running_mean->data[c] + bn->momentum * mean->data[c];

            // Update running variance using unbiased variance (num_elements - 1 for unbiased)
            float unbiased_var = var->data[c] * (float)num_elements / (num_elements - 1);  // unbiased correction
            bn->running_var->data[c] = (1 - bn->momentum) * bn->running_var->data[c] + bn->momentum * unbiased_var;
        }
    }
    else 
    {
        // Use running mean and variance during evaluation
        mean = tensor_clone(NULL, bn->running_mean);
        var = tensor_clone(NULL, bn->running_var);
    }

    // Reshape mean, var, gamma, and beta. Shape is {num_features, 1, 1}
    Tensor *mean_reshaped = tensor_reshape(mean, (int[]){num_features, 1, 1}, 3);
    Tensor *var_reshaped = tensor_reshape(var, (int[]){num_features, 1, 1}, 3);
    Tensor *gamma_reshaped = tensor_reshape(params->gamma, (int[]){num_features, 1, 1}, 3);
    Tensor *beta_reshaped = tensor_reshape(params->beta, (int[]){num_features, 1, 1}, 3);

    // Normalize the input: (x - mean) / sqrt(var + epsilon)
    Tensor *x_minus_mean = tensor_sub(x, mean_reshaped);
    Tensor *var_plus_epsilon = tensor_scalar_add(var_reshaped, bn->epsilon);
    Tensor *stddev = tensor_sqrt(var_plus_epsilon);
    Tensor *normalized_x = tensor_div(x_minus_mean, stddev);

    // Scale and shift: y = gamma * normalized_x + beta
    Tensor *scaled_x = tensor_mul(normalized_x, gamma_reshaped);
    Tensor *y = tensor_add(scaled_x, beta_reshaped);

    // Store intermediate tensors
    self->input = x;
    self->tensor_count = 12;
    self->tensors = (Tensor **)malloc(self->tensor_count * sizeof(Tensor *));
    self->tensors[0] = mean;
    self->tensors[1] = var;
    self->tensors[2] = mean_reshaped;
    self->tensors[3] = var_reshaped;
    self->tensors[4] = gamma_reshaped;
    self->tensors[5] = beta_reshaped;
    self->tensors[6] = x_minus_mean;
    self->tensors[7] = var_plus_epsilon;
    self->tensors[8] = stddev;
    self->tensors[9] = normalized_x;
    self->tensors[10] = scaled_x;
    self->tensors[11] = y;
    self->output = y;

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