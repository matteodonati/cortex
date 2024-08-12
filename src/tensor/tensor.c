#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "tensor/tensor.h"

Tensor* tensor_from_array(float *array, int *shape, int ndim) 
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int *)malloc(ndim * sizeof(int));
    tensor->stride = (int *)malloc(ndim * sizeof(int));

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) 
    {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }
    
    tensor->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) 
    {
        tensor->stride[i] = tensor->stride[i + 1] * tensor->shape[i + 1];
    }

    tensor->data = (float *)malloc(tensor->size * sizeof(float));
    memcpy(tensor->data, array, tensor->size * sizeof(float));

    return tensor;
}

Tensor* tensor_rand(int *shape, int ndim) 
{
    srand(time(NULL));

    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int *)malloc(ndim * sizeof(int));
    tensor->stride = (int *)malloc(ndim * sizeof(int));

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) 
    {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    tensor->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) 
    {
        tensor->stride[i] = tensor->stride[i + 1] * tensor->shape[i + 1];
    }

    tensor->data = (float *)malloc(tensor->size * sizeof(float));
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = (float)rand() / RAND_MAX;
    }

    return tensor;
}

Tensor* tensor_zeros(int *shape, int ndim) 
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int *)malloc(ndim * sizeof(int));
    tensor->stride = (int *)malloc(ndim * sizeof(int));

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) 
    {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    tensor->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) 
    {
        tensor->stride[i] = tensor->stride[i + 1] * tensor->shape[i + 1];
    }

    tensor->data = (float *)calloc(tensor->size, sizeof(float));

    return tensor;
}

Tensor* tensor_ones(int *shape, int ndim) 
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int *)malloc(ndim * sizeof(int));
    tensor->stride = (int *)malloc(ndim * sizeof(int));

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) 
    {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    tensor->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) 
    {
        tensor->stride[i] = tensor->stride[i + 1] * tensor->shape[i + 1];
    }

    tensor->data = (float *)malloc(tensor->size * sizeof(float));
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = 1.0f;
    }

    return tensor;
}

Tensor* tensor_eye(int size) 
{
    int shape[2] = {size, size};
    Tensor *tensor = tensor_zeros(shape, 2);

    for (int i = 0; i < size; i++) 
    {
        tensor->data[i * size + i] = 1.0f;
    }

    return tensor;
}

void free_tensor(Tensor *tensor) 
{
    if(tensor) 
    {
        free(tensor->data);
        free(tensor->shape);
        free(tensor->stride);
        free(tensor);
    }
}