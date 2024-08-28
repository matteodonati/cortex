#include <stdlib.h>
#include <string.h>
#include "tensor/tensor.h"

Tensor* initialize_tensor(const char *name, int *shape, int ndim) 
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->name = NULL;
    if (name)
    {
        tensor->name = (char *)malloc((strlen(name) + 1) * sizeof(char));
        strcpy(tensor->name, name);
    }
    tensor->ndim = ndim;
    tensor->shape = (int *)malloc(ndim * sizeof(int));
    tensor->stride = (int *)malloc(ndim * sizeof(int));
    tensor->ops_utils.cached_axes = (int *)malloc(ndim * sizeof(int));
    tensor->frozen = 0;
    tensor->backward = NULL;
    tensor->grad_a = NULL;
    tensor->grad_b = NULL;

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

    return tensor;
}

Tensor* tensor_from_array(const char *name, float *array, int *shape, int ndim) 
{
    Tensor *tensor = initialize_tensor(name, shape, ndim);
    memcpy(tensor->data, array, tensor->size * sizeof(float));
    return tensor;
}

Tensor* tensor_rand(const char *name, int *shape, int ndim, float limit) 
{
    Tensor *tensor = initialize_tensor(name, shape, ndim);
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = 2.0f * limit * ((float)rand() / RAND_MAX) - limit;
    }
    return tensor;
}

Tensor* tensor_zeros(const char *name, int *shape, int ndim) 
{
    Tensor *tensor = initialize_tensor(name, shape, ndim);
    memset(tensor->data, 0, tensor->size * sizeof(float));
    return tensor;
}

Tensor* tensor_ones(const char *name, int *shape, int ndim) 
{
    Tensor *tensor = initialize_tensor(name, shape, ndim);
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = 1.0f;
    }
    return tensor;
}

Tensor* tensor_full(const char *name, int *shape, int ndim, float value) 
{
    Tensor *tensor = initialize_tensor(name, shape, ndim);
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = value;
    }
    return tensor;
}

Tensor* tensor_like(const char *name, Tensor *a) 
{
    return initialize_tensor(name, a->shape, a->ndim);
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
        if (tensor->ops_utils.cached_axes)
        {
            free(tensor->ops_utils.cached_axes);
        }
        if (tensor->name)
        {
            free(tensor->name);
        }
        free(tensor);
    }
}