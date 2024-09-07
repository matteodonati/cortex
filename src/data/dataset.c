#include <stdlib.h>
#include <string.h>
#include "data/dataset.h"

Dataset* dataset_create(float *x, float *y, int num_samples, int *x_shape, int x_ndim, int *y_shape, int y_ndim) 
{
    Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));
    dataset->x = x;
    dataset->y = y;
    dataset->x_shape = (int *)malloc(x_ndim * sizeof(int));
    dataset->y_shape = (int *)malloc(y_ndim * sizeof(int));
    memcpy(dataset->x_shape, x_shape, x_ndim * sizeof(int));
    memcpy(dataset->y_shape, y_shape, y_ndim * sizeof(int));
    dataset->num_samples = num_samples;
    dataset->x_ndim = x_ndim;
    dataset->y_ndim = y_ndim;
    return dataset;
}

void dataset_get_sample(Dataset *dataset, int index, Tensor **x, Tensor **y) 
{
    int x_size = 1;
    int y_size = 1;
    for (int i = 0; i < dataset->x_ndim; i++)
    {
        x_size *= dataset->x_shape[i];
    }
    for (int i = 0; i < dataset->y_ndim; i++)
    {
        y_size *= dataset->y_shape[i];
    }
    *x = tensor_from_array(NULL, &dataset->x[index * x_size], dataset->x_shape, dataset->x_ndim);
    *y = tensor_from_array(NULL, &dataset->y[index * y_size], dataset->y_shape, dataset->y_ndim);
}

void dataset_free(Dataset *dataset) 
{
    if (dataset) 
    {
        free(dataset->x_shape);
        free(dataset->y_shape);
        free(dataset);
    }
}