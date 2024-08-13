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
    tensor->grad = (float*)calloc(tensor->size, sizeof(float));
    memcpy(tensor->data, array, tensor->size * sizeof(float));

    tensor->backward = NULL;
    tensor->grad_a = NULL;
    tensor->grad_b = NULL;

    return tensor;
}

Tensor* tensor_rand(int *shape, int ndim) 
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
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = (float)rand() / RAND_MAX;
    }

    tensor->backward = NULL;
    tensor->grad_a = NULL;
    tensor->grad_b = NULL;

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
    tensor->grad = (float*)calloc(tensor->size, sizeof(float));

    tensor->backward = NULL;
    tensor->grad_a = NULL;
    tensor->grad_b = NULL;

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
    tensor->grad = (float*)calloc(tensor->size, sizeof(float));
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = 1.0;
    }

    tensor->backward = NULL;
    tensor->grad_a = NULL;
    tensor->grad_b = NULL;

    return tensor;
}

Tensor* tensor_full(int *shape, int ndim, float value) 
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
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = value;
    }

    tensor->backward = NULL;
    tensor->grad_a = NULL;
    tensor->grad_b = NULL;

    return tensor;
}

Tensor* allocate_tensor_with_same_shape(Tensor *a) 
{
    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->data = (float*)malloc(a->size * sizeof(float));
    result->grad = (float*)calloc(a->size, sizeof(float));
    result->shape = (int*)malloc(a->ndim * sizeof(int));
    result->stride = (int*)malloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->size = a->size;

    memcpy(result->shape, a->shape, a->ndim * sizeof(int));
    memcpy(result->stride, a->stride, a->ndim * sizeof(int));

    result->backward = NULL;
    result->grad_a = NULL;
    result->grad_b = NULL;

    return result;
}

void tensor_free(Tensor *tensor) 
{
    if(tensor) 
    {
        free(tensor->data);
        free(tensor->grad);
        free(tensor->shape);
        free(tensor->stride);
        free(tensor);
    }
}