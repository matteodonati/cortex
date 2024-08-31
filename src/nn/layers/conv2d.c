#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn/layers/conv2d.h"
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"

Layer *conv2d_create(const char *name, int in_channels, int out_channels, int kernel_size[2], int stride[2], int padding[2], int groups, int padding_mode) 
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
    conv2d->groups = groups;
    conv2d->padding_mode = padding_mode;

    conv2d->base.name = NULL;
    if (name) 
    {
        conv2d->base.name = (char *)malloc((strlen(name) + 1) * sizeof(char));
        strcpy(conv2d->base.name, name);
    }

    conv2d->base.layer_type = LAYER_TYPE_CONV2D;
    conv2d->base.params = conv2d_parameters_create(name, in_channels, out_channels, kernel_size, groups);
    conv2d->base.forward = &conv2d_forward;
    conv2d->base.free = &conv2d_free;

    return (Layer *)conv2d;
}

Tensor *conv2d_forward(Layer *self, Tensor *x) 
{
    Conv2D *conv2d = (Conv2D *)self;

    // Placeholder for the forward pass operation
    // This will involve padding, striding, and dilating the input tensor, followed by applying the convolutional filters

    return NULL;
}

void conv2d_free(Layer *self) 
{
    if (self) 
    {
        if (self->name) 
        {
            free(self->name);
        }
        free((Conv2D *)self);
    }
}