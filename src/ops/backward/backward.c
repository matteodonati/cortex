#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ops/backward/backward.h"

void tensor_negate_backward(Tensor *self, Tensor *grad) 
{
    Tensor *tensor = self->grad_a;

    for(int i = 0; i < grad->size; i++) 
    {
        tensor->grad[i] -= grad->data[i];
    }
}

void tensor_abs_backward(Tensor *self, Tensor *grad) 
{
    Tensor *tensor = self->grad_a;

    for(int i = 0; i < grad->size; i++) 
    {
        tensor->grad[i] += (tensor->data[i] >= 0 ? 1.0f : -1.0f) * grad->data[i];
    }
}

void tensor_sqrt_backward(Tensor *self, Tensor *grad) 
{
    Tensor *tensor = self->grad_a;

    for(int i = 0; i < grad->size; i++) 
    {
        tensor->grad[i] += 0.5f / sqrt(tensor->data[i]) * grad->data[i];
    }
}

void tensor_exp_backward(Tensor *self, Tensor *grad) 
{
    Tensor *tensor = self->grad_a;

    for(int i = 0; i < grad->size; i++) 
    {
        tensor->grad[i] += exp(tensor->data[i]) * grad->data[i];
    }
}

void tensor_add_backward(Tensor *self, Tensor *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for(int i = 0; i < grad->size; i++) 
    {
        a->grad[i] += grad->data[i];
        b->grad[i] += grad->data[i];
    }
}

void tensor_sub_backward(Tensor *self, Tensor *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for(int i = 0; i < grad->size; i++) 
    {
        a->grad[i] += grad->data[i];
        b->grad[i] -= grad->data[i];
    }
}

void tensor_mul_backward(Tensor *self, Tensor *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for(int i = 0; i < grad->size; i++) 
    {
        a->grad[i] += grad->data[i] * b->data[i];
        b->grad[i] += grad->data[i] * a->data[i];
    }
}

void tensor_div_backward(Tensor *self, Tensor *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for(int i = 0; i < grad->size; i++) 
    {
        a->grad[i] += grad->data[i] / b->data[i];
        b->grad[i] -= grad->data[i] * a->data[i] / (b->data[i] * b->data[i]);
    }
}

void tensor_dot_backward(Tensor *self, Tensor *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    int m = a->shape[0];
    int n = b->shape[1];
    int k = a->shape[1];

    // Compute gradient w.r.t. a
    for(int i = 0; i < m; i++) 
    {
        for(int l = 0; l < k; l++) 
        {
            float sum = 0.0f;
            for(int j = 0; j < n; j++)
            {
                sum += grad->data[i * n + j] * b->data[l * n + j];
            }
            a->grad[i * k + l] += sum;
        }
    }

    // Compute gradient w.r.t. b
    for(int l = 0; l < k; l++) 
    {
        for(int j = 0; j < n; j++) 
        {
            float sum = 0.0f;
            for(int i = 0; i < m; i++) 
            {
                sum += grad->data[i * n + j] * a->data[i * k + l];
            }
            b->grad[l * n + j] += sum;
        }
    }
}

void tensor_transpose_backward(Tensor *self, Tensor *grad) 
{
    Tensor *tensor = self->grad_a;
    int *axes = tensor->stride; // Assume transpose uses original stride info

    // Reverse transpose
    for(int i = 0; i < grad->size; i++) 
    {
        int old_index = 0;
        int new_index = i;

        for(int j = 0; j < grad->ndim; j++)
        {
            int axis = axes[j];
            int coord = new_index / self->stride[axis];
            new_index %= self->stride[axis];
            old_index += coord * grad->stride[j];
        }
        tensor->grad[old_index] += grad->data[i];
    }
}

void tensor_reshape_backward(Tensor *self, Tensor *grad) 
{
    Tensor *tensor = self->grad_a;

    for(int i = 0; i < grad->size; i++) 
    {
        tensor->grad[i] += grad->data[i];
    }
}