#ifndef DATA_DATASET_H
#define DATA_DATASET_H

#include "tensor/tensor.h"

typedef struct Dataset
{
    float *x;
    float *y;
    int num_samples;
    int num_features;
    int num_outputs;
} Dataset;

Dataset* dataset_create(float *x, float *y, int num_samples, int num_features, int num_outputs);
void dataset_free(Dataset *dataset);
void dataset_get_sample(Dataset *dataset, int index, Tensor **x, Tensor **y);

#endif