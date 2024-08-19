#include <stdlib.h>
#include <string.h>
#include "tensor/tensor.h"

Tensor* initialize_tensor(int *shape, int ndim) 
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
    tensor->grad = (float*)calloc(tensor->size, sizeof(float));

    tensor->backward = NULL;
    tensor->grad_a = NULL;
    tensor->grad_b = NULL;

    return tensor;
}

Tensor* tensor_from_array(float *array, int *shape, int ndim) 
{
    Tensor *tensor = initialize_tensor(shape, ndim);
    memcpy(tensor->data, array, tensor->size * sizeof(float));
    return tensor;
}

Tensor* tensor_rand(int *shape, int ndim) 
{
    Tensor *tensor = initialize_tensor(shape, ndim);
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = (float)rand() / RAND_MAX;
    }
    return tensor;
}

Tensor* tensor_zeros(int *shape, int ndim) 
{
    Tensor *tensor = initialize_tensor(shape, ndim);
    memset(tensor->data, 0, tensor->size * sizeof(float));
    return tensor;
}

Tensor* tensor_ones(int *shape, int ndim) 
{
    Tensor *tensor = initialize_tensor(shape, ndim);
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = 1.0f;
    }
    return tensor;
}

Tensor* tensor_full(int *shape, int ndim, float value) 
{
    Tensor *tensor = initialize_tensor(shape, ndim);
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = value;
    }
    return tensor;
}

Tensor* tensor_like(Tensor *a) 
{
    return initialize_tensor(a->shape, a->ndim);
}

void tensor_free(Tensor *tensor) 
{
    if (tensor) 
    {
        if (tensor->data)
        {
            free(tensor->data);
        }
        if (tensor->grad)
        {
            free(tensor->grad);
        }
        if (tensor->shape)
        {
            free(tensor->shape);
        }
        if (tensor->stride)
        {
            free(tensor->stride);
        }
        if (tensor->axes)
        {
            free(tensor->axes);
        }
        free(tensor);
    }
}