#include <stdlib.h>
#include "data/dataset.h"

Dataset* dataset_create(float *x, float *y, int num_samples, int num_features, int num_outputs) 
{
    Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));
    dataset->x = x;
    dataset->y = y;
    dataset->num_samples = num_samples;
    dataset->num_features = num_features;
    dataset->num_outputs = num_outputs;
    return dataset;
}

void dataset_free(Dataset *dataset) 
{
    if (dataset) 
    {
        free(dataset);
    }
}

void dataset_get_sample(Dataset *dataset, int index, Tensor **x, Tensor **y) 
{
    *x = tensor_from_array(NULL, &dataset->x[index * dataset->num_features], (int[]){1, dataset->num_features}, 2);
    *y = tensor_from_array(NULL, &dataset->y[index * dataset->num_outputs], (int[]){1, dataset->num_outputs}, 2);
}