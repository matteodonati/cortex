#include <string.h>
#include <stdlib.h>
#include "tensor/tensor.h"
#include "utils/memory/pool.h"

static tensor_t* tensor_create(size_t ndim, const size_t shape[]) 
{
    if (ndim == 0 || ndim > MAX_DIMS)
    {
        return NULL;
    }
    if (shape == NULL)
    {
        return NULL;
    }

    tensor_t* tensor = (tensor_t*)pool_alloc(sizeof(tensor_t));

    if (tensor == NULL)
    {
        return NULL;
    }

    size_t size = 1;
    size_t stride[MAX_DIMS];
    for (size_t i = ndim; i > 0; --i) 
    {
        stride[i - 1] = size;
        size *= shape[i - 1];
    }
    tensor->ndim = ndim;
    tensor->size = size;
    tensor->frozen = false;
    tensor->context = NULL;
    tensor->grad_a = NULL;
    tensor->grad_b = NULL;
    tensor->backward = NULL;
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    memcpy(tensor->stride, stride, ndim * sizeof(size_t));
    
    for (size_t i = ndim; i < MAX_DIMS; ++i) 
    {
        tensor->shape[i] = 1;
        tensor->stride[i] = 1;
    }

    tensor->data = (float*)pool_alloc(size * sizeof(float));
    if (tensor->data == NULL)
    {
        pool_free(tensor);
        return NULL;
    }
    memset(tensor->data, 0, size * sizeof(float));

    tensor->grad = (float*)pool_alloc(size * sizeof(float));
    if (tensor->grad == NULL)
    {
        pool_free(tensor->data);
        pool_free(tensor);
        return NULL;
    }
    memset(tensor->grad, 0, size * sizeof(float));

    return tensor;
}

tensor_status_code_t tensor_destroy(tensor_t* tensor) 
{
    if (tensor == NULL)
    {
        return TENSOR_DESTROY_FAILURE;
    }
    if (tensor->data)
    {
        if (pool_free(tensor->data) == POOL_FREE_FAILURE)
        {
            return TENSOR_DESTROY_FAILURE;
        }
    }
    if (tensor->grad)
    {
        if (pool_free(tensor->grad) == POOL_FREE_FAILURE)
        {
            return TENSOR_DESTROY_FAILURE;
        }
    }
    if (pool_free(tensor) == POOL_FREE_FAILURE)
    {
        return TENSOR_DESTROY_FAILURE;
    }
    return TENSOR_DESTROY_SUCCESS;
}

tensor_t* tensor_from_array(const float* array, const size_t* shape, size_t ndim) 
{
    tensor_t *tensor = tensor_create(ndim, shape);
    if (tensor == NULL)
    {
        return NULL;
    }
    memcpy(tensor->data, array, tensor->size * sizeof(float));
    return tensor;
}

tensor_t* tensor_rand(const size_t* shape, size_t ndim, float limit) 
{
    tensor_t *tensor = tensor_create(ndim, shape);
    if (tensor == NULL)
    {
        return NULL;
    }
    for (size_t i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = 2.0f * limit * ((float)rand() / RAND_MAX) - limit;
    }
    return tensor;
}

tensor_t* tensor_full(const size_t* shape, size_t ndim, float value) 
{
    tensor_t *tensor = tensor_create(ndim, shape);
    if (tensor == NULL)
    {
        return NULL;
    }
    for (size_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = value;
    }
    return tensor;
}

tensor_t* tensor_zeros(const size_t* shape, size_t ndim) 
{
    tensor_t *tensor = tensor_create(ndim, shape);
    if (tensor == NULL)
    {
        return NULL;
    }
    return tensor;
}

tensor_t* tensor_ones(const size_t* shape, size_t ndim) 
{
    return tensor_full(shape, ndim, 1.0f);
}

tensor_t* tensor_like(const tensor_t *tensor) 
{
    if (tensor == NULL)
    {
        return NULL;
    }
    return tensor_create(tensor->ndim, tensor->shape);
}

tensor_t* tensor_clone(const tensor_t* tensor)
{
    if (tensor == NULL)
    {
        return NULL;
    }
    tensor_t *clone = tensor_create(tensor->ndim, tensor->shape);
    if (clone == NULL)
    {
        return NULL;
    }
    size_t data_size = tensor->size * sizeof(float);
    memcpy(clone->data, tensor->data, data_size);
    memcpy(clone->grad, tensor->grad, data_size);

    return clone;
}