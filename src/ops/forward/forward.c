#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"

Tensor* tensor_negate(Tensor *tensor) 
{
    Tensor *result = allocate_tensor_with_same_shape(tensor);

    for(int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = -tensor->data[i];
    }

    result->backward = &tensor_negate_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_abs(Tensor *tensor) 
{
    Tensor *result = allocate_tensor_with_same_shape(tensor);

    for(int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = fabs(tensor->data[i]);
    }

    result->backward = &tensor_abs_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_sqrt(Tensor *tensor) 
{
    Tensor *result = allocate_tensor_with_same_shape(tensor);

    for(int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = sqrt(tensor->data[i]);
    }

    result->backward = &tensor_sqrt_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_exp(Tensor *tensor) 
{
    Tensor *result = allocate_tensor_with_same_shape(tensor);

    for(int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = exp(tensor->data[i]);
    }

    result->backward = &tensor_exp_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_add(Tensor *a, Tensor *b) 
{
    Tensor *result = allocate_tensor_with_same_shape(a);

    for(int i = 0; i < a->size; i++) 
    {
        result->data[i] = a->data[i] + b->data[i];
    }

    result->backward = &tensor_add_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_sub(Tensor *a, Tensor *b) 
{
    Tensor *result = allocate_tensor_with_same_shape(a);

    for(int i = 0; i < a->size; i++) 
    {
        result->data[i] = a->data[i] - b->data[i];
    }

    result->backward = &tensor_sub_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_mul(Tensor *a, Tensor *b) 
{
    Tensor *result = allocate_tensor_with_same_shape(a);

    for(int i = 0; i < a->size; i++) 
    {
        result->data[i] = a->data[i] * b->data[i];
    }

    result->backward = &tensor_mul_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_div(Tensor *a, Tensor *b) 
{
    Tensor *result = allocate_tensor_with_same_shape(a);

    for(int i = 0; i < a->size; i++) 
    {
        result->data[i] = a->data[i] / b->data[i];
    }

    result->backward = &tensor_div_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_dot(Tensor *a, Tensor *b) 
{
    int m = a->shape[0];
    int n = b->shape[1];
    int k = a->shape[1];  // Assumes a->shape[1] == b->shape[0]

    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->shape = (int*)malloc(2 * sizeof(int));
    result->shape[0] = m;
    result->shape[1] = n;
    result->ndim = 2;
    result->size = m * n;
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)calloc(result->size, sizeof(float));

    for(int i = 0; i < m; i++) 
    {
        for(int j = 0; j < n; j++) 
        {
            float sum = 0.0f;
            for(int l = 0; l < k; l++) 
            {
                sum += a->data[i * k + l] * b->data[l * n + j];
            }
            result->data[i * n + j] = sum;
        }
    }

    result->backward = &tensor_dot_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_transpose(Tensor *tensor, int *axes) 
{
    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->ndim = tensor->ndim;
    result->shape = (int*)malloc(result->ndim * sizeof(int));
    result->stride = (int*)malloc(result->ndim * sizeof(int));
    result->size = tensor->size;
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)calloc(result->size, sizeof(float));

    for(int i = 0; i < result->ndim; i++) 
    {
        result->shape[i] = tensor->shape[axes[i]];
        result->stride[i] = tensor->stride[axes[i]];
    }

    for(int i = 0; i < tensor->size; i++) 
    {
        int old_index = 0;
        int new_index = i;

        for(int j = 0; j < result->ndim; j++) 
        {
            int axis = axes[j];
            int coord = new_index / tensor->stride[axis];
            new_index %= tensor->stride[axis];
            old_index += coord * tensor->stride[j];
        }
        result->data[i] = tensor->data[old_index];
    }

    result->backward = &tensor_transpose_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_reshape(Tensor *tensor, int *new_shape, int new_ndim) 
{
    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    
    result->ndim = new_ndim;
    result->shape = (int*)malloc(new_ndim * sizeof(int));
    result->stride = (int*)malloc(new_ndim * sizeof(int));

    result->size = 1;
    for (int i = 0; i < new_ndim; i++) {
        result->shape[i] = new_shape[i];
        result->size *= new_shape[i];
    }

    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)malloc(result->size * sizeof(float));
    memcpy(result->data, tensor->data, tensor->size * sizeof(float));
    memcpy(result->grad, tensor->grad, tensor->size * sizeof(float));

    result->stride[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

    result->backward = &tensor_reshape_backward;
    result->grad_a = tensor;

    return result;
}
