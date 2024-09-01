#include <math.h>
#include "ops/utils/utils.h"
#include "nn/activations/forward/forward.h"
#include "nn/activations/backward/backward.h"

void relu_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += tensor->data[i] > 0 ? self->grad[i] : 0;
    }

    backward(tensor);
}

void sigmoid_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        float sigmoid_val = 1.0 / (1.0 + exp(-tensor->data[i]));
        tensor->grad[i] += self->grad[i] * sigmoid_val * (1.0 - sigmoid_val);
    }

    backward(tensor);
}

void tanh_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        float tanh_val = tanh(tensor->data[i]);
        tensor->grad[i] += self->grad[i] * (1.0 - tanh_val * tanh_val);
    }

    backward(tensor);
}

void leaky_relu_backward(Tensor *self)
{
    Tensor *tensor = self->grad_a;
    float alpha = self->ops_utils.cached_float;
    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += tensor->data[i] > 0 ? self->grad[i] : alpha * self->grad[i];
    }
    backward(tensor);
}

void elu_backward(Tensor *self)
{
    Tensor *tensor = self->grad_a;
    float alpha = self->ops_utils.cached_float;
    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += tensor->data[i] > 0 ? self->grad[i] : alpha * exp(tensor->data[i]) * self->grad[i];
    }
    backward(tensor);
}

void gelu_backward(Tensor *self)
{
    Tensor *tensor = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        float x = tensor->data[i];
        float tanh_out = tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3)));
        float gelu_grad = 0.5 * tanh_out + (0.5 * x * (1 - tanh_out * tanh_out) * (sqrt(2 / M_PI) * (1 + 3 * 0.044715 * pow(x, 2))));
        tensor->grad[i] += gelu_grad * self->grad[i];
    }
    backward(tensor);
}

void softmax_backward(Tensor *self) 
{
    Tensor *input_tensor = self->grad_a;
    int axis = self->ops_utils.cached_int;

    // Retrieve the cached softmax output from the forward pass
    Tensor *softmax_output = input_tensor->ops_utils.cached_tensor;

    int outer_size = 1;
    int inner_size = 1;
    int axis_size = input_tensor->shape[axis];

    // Calculate the outer and inner sizes for the loop structure
    for (int i = 0; i < axis; i++) 
    {
        outer_size *= input_tensor->shape[i];
    }
    for (int i = axis + 1; i < input_tensor->ndim; i++) 
    {
        inner_size *= input_tensor->shape[i];
    }

    // Iterate over all elements in the tensor
    for (int outer = 0; outer < outer_size; outer++) 
    {
        for (int inner = 0; inner < inner_size; inner++) 
        {
            for (int i = 0; i < axis_size; i++) 
            {
                int idx_i = outer * axis_size * inner_size + i * inner_size + inner;
                float sum = 0.0f;

                // Compute the gradient for the current element
                for (int j = 0; j < axis_size; j++) 
                {
                    int idx_j = outer * axis_size * inner_size + j * inner_size + inner;
                    if (i == j) 
                    {
                        // Diagonal elements of the Jacobian matrix
                        sum += softmax_output->data[idx_i] * (1.0f - softmax_output->data[idx_j]) * self->grad[idx_j];
                    } 
                    else 
                    {
                        // Off-diagonal elements of the Jacobian matrix
                        sum -= softmax_output->data[idx_i] * softmax_output->data[idx_j] * self->grad[idx_j];
                    }
                }

                // Accumulate the computed gradient
                input_tensor->grad[idx_i] += sum;
            }
        }
    }

    backward(input_tensor);
}