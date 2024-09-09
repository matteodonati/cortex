#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops/utils/utils.h"
#include "nn/layers/conv2d.h"
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"
#include "nn/parameters/conv2d.h"

Layer *conv2d_create(const char *name, int in_channels, int out_channels, int kernel_size[2], int stride[2], int padding[2]) 
{
    Conv2D *conv2d = (Conv2D *)malloc(sizeof(Conv2D));
    conv2d->kernel_size[0] = kernel_size[0];
    conv2d->kernel_size[1] = kernel_size[1];
    conv2d->stride[0] = stride[0];
    conv2d->stride[1] = stride[1];
    conv2d->padding[0] = padding[0];
    conv2d->padding[1] = padding[1];
    conv2d->in_channels = in_channels;
    conv2d->out_channels = out_channels;
    conv2d->groups = 1; // groups greater than one is not supported for now
    conv2d->base.name = NULL;
    if (name) 
    {
        conv2d->base.name = (char *)malloc((strlen(name) + 1) * sizeof(char));
        strcpy(conv2d->base.name, name);
    }
    conv2d->base.layer_type = LAYER_TYPE_CONV2D;
    conv2d->base.params = conv2d_parameters_create(name, in_channels, out_channels, kernel_size);
    conv2d->base.is_training = false;
    conv2d->base.forward = &conv2d_forward;
    conv2d->base.free = &conv2d_free;
    return (Layer *)conv2d;
}

Tensor* conv2d_forward(Layer *self, Tensor *x) 
{
    Conv2DParameters *params = (Conv2DParameters *)self->params;
    Conv2D *layer = (Conv2D *)self;

    // Convolution parameters
    int kernel_height = layer->kernel_size[0];
    int kernel_width = layer->kernel_size[1];
    int out_channels = layer->out_channels;
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

    // im2col transformation of input tensor. Resulting shape: {batch_size, in_channels * kernel_height * kernel_width, output_height * output_width}
    Tensor *ic = im2col(x, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);

    // Reshape kernel. Shape: {out_channels, in_channels * kernel_height * kernel_width}
    int kr_ndim = 2;
    int kr_shape[] = {out_channels, in_channels * kernel_height * kernel_width};
    Tensor *kr = tensor_reshape(params->weights, kr_shape, kr_ndim);

    // Perform matrix multiplication. Shape: {batch_size, out_channels, output_height * output_width}
    Tensor *of = tensor_matmul(kr, ic);

    // Reshape bias. Shape {1, out_channels, 1}
    int br_ndim = 3;
    int br_shape[] = {1, out_channels, 1};
    Tensor *br = tensor_reshape(params->bias, br_shape, br_ndim);

    // Add bias
    Tensor *ob = tensor_add(of, br);

    // Reshape output to desired shape: {batch_size, out_channels, output_height, output_width}
    int or_ndim = 4;
    int or_shape[] = {batch_size, out_channels, output_height, output_width};
    Tensor *or = tensor_reshape(ob, or_shape, or_ndim);

    // Pointers to intermediate results
    self->input = x;
    self->tensor_count = 6;
    self->tensors = (Tensor **)malloc(self->tensor_count * sizeof(Tensor *));
    self->tensors[0] = ic;
    self->tensors[1] = kr;
    self->tensors[2] = of;
    self->tensors[3] = br;
    self->tensors[4] = ob;
    self->tensors[5] = or;
    self->output = or;

    return or;
}

void conv2d_free(Layer *self) 
{
    if (self) 
    {
        free((Conv2D *)self);
    }
}