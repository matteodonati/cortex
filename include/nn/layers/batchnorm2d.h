#ifndef NN_LAYER_BATCHNORM2D_H
#define NN_LAYER_BATCHNORM2D_H

#include "nn/layers/layer.h"

typedef struct BatchNorm2D 
{
    Layer base;
    int num_features;
    float epsilon;
    float momentum;
    Tensor *running_mean;
    Tensor *running_var;
} BatchNorm2D;

Layer* batchnorm2d_create(const char *name, int num_features, float epsilon, float momentum);
Tensor* batchnorm2d_forward(Layer *self, Tensor *x);
void batchnorm2d_free(Layer *self);

#endif