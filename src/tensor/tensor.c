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

tensor_t* tensor_create(size_t ndim, const size_t shape[], const tensor_type_t type) 
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

    tensor->type = type;
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

    void *data = NULL;
    switch (type)
    {
        case TENSOR_TYPE_FLOAT:
            data = (float*)pool_alloc(size * sizeof(float));
            tensor->data.f_data = data;
            break;
        
        case TENSOR_TYPE_UINT8:
            data = (uint8_t*)pool_alloc(size * sizeof(uint8_t));
            tensor->data.u8_data = data;
            break;
        
        default:
            return NULL;
    }
    if (data == NULL)
    {
        return NULL;
    }

    tensor->grad = NULL;
    if (type == TENSOR_TYPE_FLOAT)
    {
        tensor->grad = (float*)pool_alloc(size * sizeof(float));
        if (tensor->grad == NULL)
        {
            return NULL;
        }
        memset(tensor->grad, 0, size * sizeof(float));
    }

    return tensor;
}

tensor_status_code_t tensor_destroy(tensor_t* tensor) 
{
    if (tensor == NULL)
    {
        return TENSOR_DESTROY_FAILURE;
    }
    switch (tensor->type)
    {
        case TENSOR_TYPE_FLOAT:
            pool_free(tensor->data.f_data);
            pool_free(tensor->grad);
            break;

        case TENSOR_TYPE_UINT8:
            pool_free(tensor->data.u8_data);
    }
    pool_free(tensor);
    return TENSOR_DESTROY_SUCCESS;
}