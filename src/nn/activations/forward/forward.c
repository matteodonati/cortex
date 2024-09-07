#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn/activations/forward/forward.h"
#include "nn/activations/backward/backward.h"

Tensor* relu_f(Tensor *tensor) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in relu_f.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = tensor->data[i] > 0 ? tensor->data[i] : 0;
    }
    result->backward = &relu_backward;
    result->grad_a = tensor;
    return result;
}

Tensor* sigmoid_f(Tensor *tensor) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in sigmoid_f.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = 1.0 / (1.0 + expf(-tensor->data[i]));
    }
    result->backward = &sigmoid_backward;
    result->grad_a = tensor;
    return result;
}

Tensor* tanh_f(Tensor *tensor) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tanh_f.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = tanhf(tensor->data[i]);
    }
    result->backward = &tanh_backward;
    result->grad_a = tensor;
    return result;
}

Tensor* leaky_relu_f(Tensor *tensor, float alpha)
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in leaky_relu_f.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = tensor->data[i] > 0 ? tensor->data[i] : alpha * tensor->data[i];
    }
    result->ops_utils.cached_float = alpha;
    result->backward = &leaky_relu_backward;
    result->grad_a = tensor;
    return result;
}

Tensor* elu_f(Tensor *tensor, float alpha)
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in elu_f.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = tensor->data[i] > 0 ? tensor->data[i] : alpha * (expf(tensor->data[i]) - 1);
    }
    result->ops_utils.cached_float = alpha;
    result->backward = &elu_backward;
    result->grad_a = tensor;
    return result;
}

Tensor* gelu_f(Tensor *tensor)
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in gelu_f.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        float x = tensor->data[i];
        result->data[i] = 0.5 * x * (1 + tanhf(sqrtf(2 / M_PI) * (x + 0.044715 * powf(x, 3))));
    }
    result->backward = &gelu_backward;
    result->grad_a = tensor;
    return result;
}

Tensor* softmax_f(Tensor *tensor, int axis) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in softmax_f.\n");
        exit(EXIT_FAILURE);
    }

    if (axis < 0 || axis >= tensor->ndim)
    {
        fprintf(stderr, "Error: Axis out of bounds for input tensor in softmax_f.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);

    int outer_size = 1;
    int inner_size = 1;

    // Calculate sizes
    for (int i = 0; i < axis; i++) 
    {
        outer_size *= tensor->shape[i];
    }
    for (int i = axis + 1; i < tensor->ndim; i++) 
    {
        inner_size *= tensor->shape[i];
    }

    int axis_size = tensor->shape[axis];

    // Perform softmax along the specified axis
    for (int outer = 0; outer < outer_size; outer++) 
    {
        for (int inner = 0; inner < inner_size; inner++) 
        {
            float max_val = tensor->data[outer * axis_size * inner_size + inner];
            for (int i = 1; i < axis_size; i++) 
            {
                int idx = outer * axis_size * inner_size + i * inner_size + inner;
                if (tensor->data[idx] > max_val) 
                {
                    max_val = tensor->data[idx];
                }
            }

            float sum_exp = 0.0;
            for (int i = 0; i < axis_size; i++) 
            {
                int idx = outer * axis_size * inner_size + i * inner_size + inner;
                result->data[idx] = expf(tensor->data[idx] - max_val);
                sum_exp += result->data[idx];
            }

            for (int i = 0; i < axis_size; i++) 
            {
                int idx = outer * axis_size * inner_size + i * inner_size + inner;
                result->data[idx] /= sum_exp;
            }
        }
    }

    result->ops_utils.cached_int = axis;
    result->backward = &softmax_backward;
    result->grad_a = tensor;
    tensor->ops_utils.cached_tensor = result;
    
    return result;
}