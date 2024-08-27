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

    tensor_backward(tensor);
}

void sigmoid_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        float sigmoid_val = 1.0 / (1.0 + exp(-tensor->data[i]));
        tensor->grad[i] += self->grad[i] * sigmoid_val * (1.0 - sigmoid_val);
    }

    tensor_backward(tensor);
}

void tanh_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    for (int i = 0; i < self->size; i++) 
    {
        float tanh_val = tanh(tensor->data[i]);
        tensor->grad[i] += self->grad[i] * (1.0 - tanh_val * tanh_val);
    }

    tensor_backward(tensor);
}

void leaky_relu_backward(Tensor *self)
{
    Tensor *tensor = self->grad_a;
    float alpha = self->ops_utils.working_scalar;
    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += tensor->data[i] > 0 ? self->grad[i] : alpha * self->grad[i];
    }
    tensor_backward(tensor);
}

void elu_backward(Tensor *self)
{
    Tensor *tensor = self->grad_a;
    float alpha = self->ops_utils.working_scalar;
    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += tensor->data[i] > 0 ? self->grad[i] : alpha * exp(tensor->data[i]) * self->grad[i];
    }
    tensor_backward(tensor);
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
    tensor_backward(tensor);
}

void softmax_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

     // Axis used in softmax_f
    int axis = self->ops_utils.working_axis;

    Tensor *softmax_output = softmax_f(tensor, axis);

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < axis; i++) 
    {
        outer_size *= tensor->shape[i];
    }
    for (int i = axis + 1; i < tensor->ndim; i++) 
    {
        inner_size *= tensor->shape[i];
    }

    int axis_size = tensor->shape[axis];

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
                        sum += softmax_output->data[idx_i] * (1.0 - softmax_output->data[idx_j]) * self->grad[idx_j];
                    } 
                    else 
                    {
                        sum -= softmax_output->data[idx_i] * softmax_output->data[idx_j] * self->grad[idx_j];
                    }
                }
                tensor->grad[idx_i] += sum;
            }
        }
    }

    tensor_backward(tensor);
    tensor_free(softmax_output);
}