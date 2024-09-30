#include <assert.h>
#include <string.h>
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"

tensor_t* tensor_add(const tensor_t* __restrict__ a, const tensor_t* __restrict__ b) 
{
    assert(a != NULL && "First input tensor is NULL in tensor_add.");
    assert(b != NULL && "Second input tensor is NULL in tensor_add.");
    assert(a->size == b->size && "Tensors must have the same size in tensor_add.");

    tensor_t* result = tensor_like(a);
    assert(result != NULL && "Failed to allocate memory for result tensor in tensor_add.");

    size_t size = a->size;
    const float* __restrict__ a_data = a->data;
    const float* __restrict__ b_data = b->data;
    float* __restrict__ res_data = result->data;

    for (size_t i = 0; i < size; ++i) 
    {
        res_data[i] = a_data[i] + b_data[i];
    }

    result->backward = tensor_add_backward;
    result->grad_a = (tensor_t*)a;
    result->grad_b = (tensor_t*)b;

    return result;
}

tensor_t* tensor_reshape(const tensor_t* __restrict__ tensor, const size_t* __restrict__ new_shape, size_t new_ndim) 
{
    assert(tensor != NULL && "Input tensor is NULL in tensor_reshape.");
    assert(new_shape != NULL && "New shape is NULL in tensor_reshape.");
    assert(new_ndim > 0 && "New number of dimensions must be positive in tensor_reshape.");

    tensor_t *result = tensor_zeros(new_shape, new_ndim);
    assert(result != NULL && "Failed to allocate memory for result tensor in tensor_reshape.");

    memcpy(result->data, tensor->data, tensor->size * sizeof(float));
    memcpy(result->grad, tensor->grad, tensor->size * sizeof(float));

    result->backward = tensor_reshape_backward;
    result->grad_a = (tensor_t*)tensor;

    return result;
}