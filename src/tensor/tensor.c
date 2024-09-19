#include <string.h>
#include "tensor/tensor.h"
#include "utils/memory/pool.h"

static void compute_size_and_stride(size_t ndim, const size_t shape[], size_t stride[], size_t* size) 
{
    *size = 1;
    for (size_t i = ndim; i > 0; --i) 
    {
        stride[i - 1] = *size;
        *size *= shape[i - 1];
    }
}

tensor_t* tensor_create(size_t ndim, const size_t shape[]) 
{
    if (ndim <= 0 || ndim > MAX_DIMS)
    {
        return NULL;
    }
    if (shape == NULL)
    {
        return NULL;
    }

    size_t size;
    size_t stride[MAX_DIMS];

    compute_size_and_stride(ndim, shape, stride, &size);

    tensor_t* tensor = (tensor_t*)pool_alloc(sizeof(tensor_t));
    if (tensor == NULL)
    {
        return NULL;
    }

    tensor->data = (float*)pool_alloc(size * sizeof(float));
    if (tensor->data == NULL)
    {
        return NULL;
    }

    tensor->grad = (float*)pool_alloc(size * sizeof(float));
    if (tensor->grad == NULL)
    {
        return NULL;
    }
    memset(tensor->grad, 0, size * sizeof(float));

    tensor->ndim = ndim;
    tensor->size = size;
    tensor->frozen = false;
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

    return tensor;
}

tensor_status_code_t tensor_destroy(tensor_t* tensor) 
{
    if (tensor == NULL)
    {
        return TENSOR_DESTROY_FAILURE;
    }
    pool_free(tensor->grad);
    pool_free(tensor->data);
    pool_free(tensor);
    return TENSOR_DESTROY_SUCCESS;
}