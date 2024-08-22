#include <math.h>
#include <string.h>
#include "nn/activations/forward/forward.h"
#include "nn/activations/backward/backward.h"

Tensor* relu_f(Tensor *input) 
{
    Tensor *result = tensor_like(NULL, input);
    for (int i = 0; i < input->size; i++) 
    {
        result->data[i] = input->data[i] > 0 ? input->data[i] : 0;
    }
    result->backward = &relu_backward;
    result->grad_a = input;
    return result;
}

Tensor* sigmoid_f(Tensor *input) 
{
    Tensor *result = tensor_like(NULL, input);
    for (int i = 0; i < input->size; i++) 
    {
        result->data[i] = 1.0 / (1.0 + exp(-input->data[i]));
    }
    result->backward = &sigmoid_backward;
    result->grad_a = input;
    return result;
}

Tensor* tanh_f(Tensor *input) 
{
    Tensor *result = tensor_like(NULL, input);
    for (int i = 0; i < input->size; i++) 
    {
        result->data[i] = tanh(input->data[i]);
    }
    result->backward = &tanh_backward;
    result->grad_a = input;
    return result;
}

Tensor* softmax_f(Tensor *input, int axis) 
{
    Tensor *result = tensor_like(NULL, input);

    int outer_size = 1;
    int inner_size = 1;

    // Calculate sizes
    for (int i = 0; i < axis; i++) 
    {
        outer_size *= input->shape[i];
    }
    for (int i = axis + 1; i < input->ndim; i++) 
    {
        inner_size *= input->shape[i];
    }

    int axis_size = input->shape[axis];

    // Perform softmax along the specified axis
    for (int outer = 0; outer < outer_size; outer++) 
    {
        for (int inner = 0; inner < inner_size; inner++) 
        {
            float max_val = input->data[outer * axis_size * inner_size + inner];
            for (int i = 1; i < axis_size; i++) 
            {
                int idx = outer * axis_size * inner_size + i * inner_size + inner;
                if (input->data[idx] > max_val) 
                {
                    max_val = input->data[idx];
                }
            }

            float sum_exp = 0.0;
            for (int i = 0; i < axis_size; i++) 
            {
                int idx = outer * axis_size * inner_size + i * inner_size + inner;
                result->data[idx] = exp(input->data[idx] - max_val);
                sum_exp += result->data[idx];
            }

            for (int i = 0; i < axis_size; i++) 
            {
                int idx = outer * axis_size * inner_size + i * inner_size + inner;
                result->data[idx] /= sum_exp;
            }
        }
    }

    result->ops_utils.working_axis = axis;
    result->backward = &softmax_backward;
    result->grad_a = input;
    
    return result;
}