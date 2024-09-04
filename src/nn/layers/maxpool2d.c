#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn/layers/maxpool2d.h"
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"

Layer* maxpool2d_create(const char *name, int kernel_size[2], int stride[2], int padding[2]) 
{
    MaxPool2D *pool = (MaxPool2D *)malloc(sizeof(MaxPool2D));

    pool->kernel_size[0] = kernel_size[0];
    pool->kernel_size[1] = kernel_size[1];
    pool->stride[0] = stride[0];
    pool->stride[1] = stride[1];
    pool->padding[0] = padding[0];
    pool->padding[1] = padding[1];
    
    pool->base.name = NULL;
    if (name) 
    {
        pool->base.name = (char *)malloc((strlen(name) + 1) * sizeof(char));
        strcpy(pool->base.name, name);
    }
    pool->base.layer_type = LAYER_TYPE_MAXPOOL;
    pool->base.params = NULL;
    pool->base.forward = &maxpool2d_forward;
    pool->base.free = &maxpool2d_free;

    return (Layer *)pool;
}

Tensor* maxpool2d_forward(Layer *self, Tensor *x) 
{
    MaxPool2D *layer = (MaxPool2D *)self;

    // Pooling parameters
    int kernel_height = layer->kernel_size[0];
    int kernel_width = layer->kernel_size[1];
    int stride_height = layer->stride[0];
    int stride_width = layer->stride[1];
    int pad_height = layer->padding[0];
    int pad_width = layer->padding[1];

    // Input information
    int batch_size = x->shape[0];
    int in_channels = x->shape[1];
    int input_height = x->shape[2];
    int input_width = x->shape[3];

    // Calculate output dimensions
    int output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

    // im2col transformation of input tensor. Shape: {batch_size, in_channels * kernel_height * kernel_width, output_height * output_width}
    Tensor *input_col = im2col(x, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);

    // Reshape input_col to {batch_size, in_channels, kernel_height * kernel_width, output_height * output_width}
    Tensor *input_col_reshaped = tensor_reshape(input_col, (int[]){batch_size, in_channels, kernel_height * kernel_width, output_height * output_width}, 4);

    // Perform max operation along the kernel size axis (axis 2)
    Tensor *output_col = tensor_max(input_col_reshaped, 2);

    // Reshape the output back to {batch_size, in_channels, output_height, output_width}
    Tensor *output_reshaped = tensor_reshape(output_col, (int[]){batch_size, in_channels, output_height, output_width}, 4);

    // Store intermediate results for backpropagation
    self->input = x;
    self->tensor_count = 4;
    self->tensors = (Tensor **)malloc(self->tensor_count * sizeof(Tensor *));
    self->tensors[0] = input_col;
    self->tensors[1] = input_col_reshaped;
    self->tensors[2] = output_col;
    self->tensors[3] = output_reshaped;
    self->output = output_reshaped;

    return output_reshaped;
}

void maxpool2d_free(Layer *self) 
{
    if (self) 
    {
        free((MaxPool2D *)self);
    }
}