#include <string.h> 
#include "ops/backward/backward.h"

void tensor_add_backward(tensor_t* self) 
{
    if (self == NULL || self->grad_a == NULL || self->grad_b == NULL)
    {
        return;
    }

    tensor_t* a = self->grad_a;
    tensor_t* b = self->grad_b;
    float* a_grad = a->grad;
    float* b_grad = b->grad;
    const float* grad_output = self->grad;

    for (size_t i = 0; i < self->size; ++i) 
    {
        a_grad[i] += grad_output[i];
        b_grad[i] += grad_output[i];
    }

    tensor_backward(a);
    tensor_backward(b);
}

void tensor_reshape_backward(tensor_t* self) 
{
    if (self == NULL || self->grad_a == NULL)
    {
        return;
    }

    tensor_t* tensor = self->grad_a;
    float* grad = tensor->grad;
    const float* self_grad = self->grad;

    for (size_t i = 0; i < self->size; ++i) 
    {
        grad[i] += self_grad[i];
    }

    tensor_backward(tensor);
}