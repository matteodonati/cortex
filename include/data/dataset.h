#ifndef DATA_DATASET_H
#define DATA_DATASET_H

#include "tensor/tensor.h"

typedef struct Dataset
{
    float *x;
    float *y;
    int *x_shape;
    int *y_shape;
    int num_samples;
    int x_ndim;
    int y_ndim;
} Dataset;

Dataset* dataset_create(float *x, float *y, int num_samples, int *x_shape, int x_ndim, int *y_shape, int y_ndim);
void dataset_get_sample(Dataset *dataset, int index, Tensor **x, Tensor **y);
void dataset_free(Dataset *dataset);

#endif