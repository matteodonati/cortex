#include <string.h>
#include <stdlib.h>
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

static tensor_t* tensor_create(size_t ndim, const size_t shape[], tensor_type_t type) 
{
    if (ndim == 0 || ndim > MAX_DIMS)
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

    switch (type)
    {
        case TENSOR_TYPE_FLOAT:
            tensor->data.f_data = (float*)pool_alloc(size * sizeof(float));
            if (tensor->data.f_data == NULL)
            {
                pool_free(tensor);
                return NULL;
            }
            memset(tensor->data.f_data, 0, size * sizeof(float));

            tensor->grad = (float*)pool_alloc(size * sizeof(float));
            if (tensor->grad == NULL)
            {
                pool_free(tensor->data.f_data);
                pool_free(tensor);
                return NULL;
            }
            memset(tensor->grad, 0, size * sizeof(float));
            break;
        
        case TENSOR_TYPE_UINT8:
            tensor->data.u8_data = (uint8_t*)pool_alloc(size * sizeof(uint8_t));
            if (tensor->data.u8_data == NULL)
            {
                pool_free(tensor);
                return NULL;
            }
            memset(tensor->data.f_data, 0, size * sizeof(float));
            tensor->grad = NULL;
            break;
        
        default:
            pool_free(tensor);
            return NULL;
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
            break;
    }
    pool_free(tensor);
    return TENSOR_DESTROY_SUCCESS;
}

tensor_t* tensor_from_array(const float *array, const size_t *shape, size_t ndim) 
{
    tensor_t *tensor = tensor_create(ndim, shape, TENSOR_TYPE_FLOAT);
    if (tensor == NULL)
    {
        return NULL;
    }
    memcpy(tensor->data.f_data, array, tensor->size * sizeof(float));
    return tensor;
}

tensor_t* tensor_rand(const size_t *shape, size_t ndim, float limit) 
{
    tensor_t *tensor = tensor_create(ndim, shape, TENSOR_TYPE_FLOAT);
    if (tensor == NULL)
    {
        return NULL;
    }
    for (size_t i = 0; i < tensor->size; i++) 
    {
        tensor->data.f_data[i] = 2.0f * limit * ((float)rand() / RAND_MAX) - limit;
    }
    return tensor;
}

tensor_t* tensor_full(const size_t *shape, size_t ndim, float value) 
{
    tensor_t *tensor = tensor_create(ndim, shape, TENSOR_TYPE_FLOAT);
    if (tensor == NULL)
    {
        return NULL;
    }
    for (size_t i = 0; i < tensor->size; i++)
    {
        tensor->data.f_data[i] = value;
    }
    return tensor;
}

tensor_t* tensor_zeros(const size_t *shape, size_t ndim) 
{
    return tensor_full(shape, ndim, 0.0f);
}

tensor_t* tensor_ones(const size_t *shape, size_t ndim) 
{
    return tensor_full(shape, ndim, 1.0f);
}

tensor_t* tensor_like(const tensor_t *tensor) 
{
    if (tensor == NULL)
    {
        return NULL;
    }
    return tensor_create(tensor->ndim, tensor->shape, tensor->type);
}

tensor_t* tensor_clone(const tensor_t *tensor)
{
    if (tensor == NULL)
    {
        return NULL;
    }
    tensor_t *clone = tensor_create(tensor->ndim, tensor->shape, tensor->type);
    if (clone == NULL)
    {
        return NULL;
    }
    size_t data_size = tensor->size * sizeof(float);
    memcpy(clone->data.f_data, tensor->data.f_data, data_size);
    memcpy(clone->grad, tensor->grad, data_size);

    return clone;
}