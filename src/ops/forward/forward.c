#include <string.h>
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"

tensor_t* tensor_add(const tensor_t* a, const tensor_t* b) 
{
    if (a == NULL || b == NULL)
    {
        return NULL;
    }
    if (a->size != b->size)
    {
        return NULL;
    }

    tensor_t* result = tensor_like(a);
    if (result == NULL)
    {
        return NULL;
    }

    size_t size = a->size;
    const float* a_data = a->data;
    const float* b_data = b->data;
    float* res_data = result->data;

    for (size_t i = 0; i < size; ++i) 
    {
        res_data[i] = a_data[i] + b_data[i];
    }

    result->backward = tensor_add_backward;
    result->grad_a = (tensor_t*)a;
    result->grad_b = (tensor_t*)b;

    return result;
}

tensor_t* tensor_reshape(const tensor_t* tensor, const size_t* new_shape, size_t new_ndim) 
{
    if (tensor == NULL || new_shape == NULL)
    {
        return NULL;
    }
    if (new_ndim == 0)
    {
        return NULL;
    }

    tensor_t *result = tensor_zeros(new_shape, new_ndim);
    if (result == NULL)
    {
        return NULL;
    }

    memcpy(result->data, tensor->data, tensor->size * sizeof(float));
    memcpy(result->grad, tensor->grad, tensor->size * sizeof(float));

    result->backward = tensor_reshape_backward;
    result->grad_a = (tensor_t*)tensor;

    return result;
}