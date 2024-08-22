#include <math.h>
#include "nn/activations/forward/forward.h"
#include "nn/activations/backward/backward.h"

void relu_backward(Tensor *self, float *grad) 
{
    Tensor *input = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        input->grad[i] += input->data[i] > 0 ? grad[i] : 0;
    }

    if (input->backward) 
    {
        input->backward(input, input->grad);
    }
}

void sigmoid_backward(Tensor *self, float *grad) 
{
    Tensor *input = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        float sigmoid_val = 1.0 / (1.0 + exp(-input->data[i]));
        input->grad[i] += grad[i] * sigmoid_val * (1.0 - sigmoid_val);
    }

    if (input->backward) 
    {
        input->backward(input, input->grad);
    }
}

void tanh_backward(Tensor *self, float *grad) 
{
    Tensor *input = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        float tanh_val = tanh(input->data[i]);
        input->grad[i] += grad[i] * (1.0 - tanh_val * tanh_val);
    }

    if (input->backward) 
    {
        input->backward(input, input->grad);
    }
}

void softmax_backward(Tensor *self, float *grad) 
{
    Tensor *input = self->grad_a;

     // Axis used in softmax_f
    int axis = self->ops_utils.working_axis;

    Tensor *softmax_output = softmax_f(input, axis);

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < axis; i++) 
    {
        outer_size *= input->shape[i];
    }
    for (int i = axis + 1; i < input->ndim; i++) 
    {
        inner_size *= input->shape[i];
    }

    int axis_size = input->shape[axis];

    for (int outer = 0; outer < outer_size; outer++) 
    {
        for (int inner = 0; inner < inner_size; inner++) 
        {
            for (int i = 0; i < axis_size; i++) 
            {
                int idx_i = outer * axis_size * inner_size + i * inner_size + inner;
                float sum = 0.0;
                for (int j = 0; j < axis_size; j++) 
                {
                    int idx_j = outer * axis_size * inner_size + j * inner_size + inner;
                    if (i == j) 
                    {
                        sum += softmax_output->data[idx_i] * (1.0 - softmax_output->data[idx_j]) * grad[idx_j];
                    } 
                    else 
                    {
                        sum -= softmax_output->data[idx_i] * softmax_output->data[idx_j] * grad[idx_j];
                    }
                }
                input->grad[idx_i] += sum;
            }
        }
    }

    if (input->backward) 
    {
        input->backward(input, input->grad);
    }

    tensor_free(softmax_output);
}