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

    conv2d->in_channels = in_channels;
    conv2d->out_channels = out_channels;
    conv2d->kernel_size[0] = kernel_size[0];
    conv2d->kernel_size[1] = kernel_size[1];
    conv2d->stride[0] = stride[0];
    conv2d->stride[1] = stride[1];
    conv2d->padding[0] = padding[0];
    conv2d->padding[1] = padding[1];
    conv2d->groups = 1; // groups greater than one is not supported for now

    conv2d->base.name = NULL;
    if (name) 
    {
        conv2d->base.name = (char *)malloc((strlen(name) + 1) * sizeof(char));
        strcpy(conv2d->base.name, name);
    }

    conv2d->base.layer_type = LAYER_TYPE_CONV2D;
    conv2d->base.params = conv2d_parameters_create(name, in_channels, out_channels, kernel_size);
    conv2d->base.forward = &conv2d_forward;
    conv2d->base.free = &conv2d_free;

    return (Layer *)conv2d;
}

Tensor* im2col(Tensor *input, int kernel_height, int kernel_width, int stride_height, int stride_width, int pad_height, int pad_width) 
{
    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int input_height = input->shape[2];
    int input_width = input->shape[3];

    // Calculate the output dimensions
    int output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

    // The dimensions of the resulting column matrix
    int col_height = in_channels * kernel_height * kernel_width;
    int col_width = output_height * output_width;

    // Allocate memory for the column matrix
    Tensor *col = tensor_zeros(NULL, (int[]){batch_size, col_height, col_width}, 3);

    // Populate the column matrix
    for (int n = 0; n < batch_size; n++) 
    {
        for (int c = 0; c < in_channels; c++) 
        {
            for (int kh = 0; kh < kernel_height; kh++) 
            {
                for (int kw = 0; kw < kernel_width; kw++) 
                {
                    for (int oh = 0; oh < output_height; oh++) 
                    {
                        for (int ow = 0; ow < output_width; ow++) 
                        {
                            int h = oh * stride_height - pad_height + kh;
                            int w = ow * stride_width - pad_width + kw;
                            int col_index = ((c * kernel_height + kh) * kernel_width + kw) * col_width + oh * output_width + ow;
                            if (h >= 0 && h < input_height && w >= 0 && w < input_width) 
                            {
                                col->data[n * col_height * col_width + col_index] = input->data[n * in_channels * input_height * input_width + c * input_height * input_width + h * input_width + w];
                            }
                        }
                    }
                }
            }
        }
    }

    col->backward = col2im;
    col->grad_a = input;

    // Store necessary parameters for backward pass
    col->ops_utils.cached_ints = (int *)malloc(6 * sizeof(int));
    col->ops_utils.cached_ints[0] = kernel_height;
    col->ops_utils.cached_ints[1] = kernel_width;
    col->ops_utils.cached_ints[2] = stride_height;
    col->ops_utils.cached_ints[3] = stride_width;
    col->ops_utils.cached_ints[4] = pad_height;
    col->ops_utils.cached_ints[5] = pad_width;

    return col;
}

void col2im(Tensor *self) 
{
    Tensor *input = self->grad_a;

    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int input_height = input->shape[2];
    int input_width = input->shape[3];

    // Retrieve the parameters from the forward pass
    int kernel_height = self->ops_utils.cached_ints[0]; 
    int kernel_width = self->ops_utils.cached_ints[1];
    int stride_height = self->ops_utils.cached_ints[2];
    int stride_width = self->ops_utils.cached_ints[3];
    int pad_height = self->ops_utils.cached_ints[4];
    int pad_width = self->ops_utils.cached_ints[5];

    // Calculate the output dimensions
    int output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

    // Iterate over the batches and channels
    for (int n = 0; n < batch_size; n++) 
    {
        for (int c = 0; c < in_channels; c++) 
        {
            for (int kh = 0; kh < kernel_height; kh++) 
            {
                for (int kw = 0; kw < kernel_width; kw++) 
                {
                    for (int oh = 0; oh < output_height; oh++) 
                    {
                        for (int ow = 0; ow < output_width; ow++) 
                        {
                            int h = oh * stride_height - pad_height + kh;
                            int w = ow * stride_width - pad_width + kw;
                            int col_index = ((c * kernel_height + kh) * kernel_width + kw) * output_width * output_height + oh * output_width + ow;
                            if (h >= 0 && h < input_height && w >= 0 && w < input_width) 
                            {
                                input->grad[n * in_channels * input_height * input_width + c * input_height * input_width + h * input_width + w] += self->grad[n * self->shape[1] * self->shape[2] + col_index];
                            }
                        }
                    }
                }
            }
        }
    }

    backward(input);
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
    Tensor *input_col = im2col(x, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width);

    // Reshape kernel to 2D for matmul. Shape: {out_channels, in_channels * kernel_height * kernel_width}
    Tensor *kernel_reshaped = tensor_reshape(params->weights, (int[]){out_channels, in_channels * kernel_height * kernel_width}, 2);

    // Perform matrix multiplication. Shape: {batch_size, out_channels, output_height * output_width}
    Tensor *output_flat = tensor_matmul(kernel_reshaped, input_col);

    // Reshape output to desired shape: {batch_size, out_channels, output_height, output_width}
    Tensor *output_reshaped = tensor_reshape(output_flat, (int[]){batch_size, out_channels, output_height, output_width}, 4);

    // Add bias (broadcasting automatically handles the addition)
    Tensor *output_with_bias = tensor_add(output_reshaped, params->bias);

    // Set pointers to intermediate results
    self->input = x;
    self->tensor_count = 5;
    self->tensors = (Tensor **)malloc(self->tensor_count * sizeof(Tensor *));
    self->tensors[0] = input_col;
    self->tensors[1] = kernel_reshaped;
    self->tensors[2] = output_flat;
    self->tensors[3] = output_reshaped;
    self->tensors[4] = output_with_bias;
    self->output = output_with_bias;

    return output_with_bias;
}

void conv2d_free(Layer *self) 
{
    if (self) 
    {
        free((Conv2D *)self);
    }
}