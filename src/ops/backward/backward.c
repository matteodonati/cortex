#include <math.h>
#include <stdlib.h>
#include "ops/backward/backward.h"

#include <stdio.h>

void tensor_negate_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] -= grad[i];
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_abs_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += (tensor->data[i] >= 0 ? 1.0f : -1.0f) * grad[i];
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_sqrt_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += 0.5f / sqrt(tensor->data[i]) * grad[i];
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_exp_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += exp(tensor->data[i]) * grad[i];
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_log_backward(Tensor *self, float *grad)
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++)
    {
        tensor->grad[i] += grad[i] / tensor->data[i];
    }

    if (tensor->backward)
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_add_backward(Tensor *self, float *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for (int i = 0; i < self->size; i++) 
    {
        a->grad[i] += grad[i];
        b->grad[i] += grad[i];
    }

    if (a->backward) 
    {
        a->backward(a, a->grad);
    }
    if (b->backward) 
    {
        b->backward(b, b->grad);
    }
}

void tensor_sub_backward(Tensor *self, float *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for (int i = 0; i < self->size; i++) 
    {
        a->grad[i] += grad[i];
        b->grad[i] -= grad[i];
    }

    if (a->backward) 
    {
        a->backward(a, a->grad);
    }
    if (b->backward) 
    {
        b->backward(b, b->grad);
    }
}

void tensor_mul_backward(Tensor *self, float *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for (int i = 0; i < self->size; i++) 
    {
        a->grad[i] += grad[i] * b->data[i];
        b->grad[i] += grad[i] * a->data[i];
    }

    if (a->backward) 
    {
        a->backward(a, a->grad);
    }
    if (b->backward) 
    {
        b->backward(b, b->grad);
    }
}

void tensor_div_backward(Tensor *self, float *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for (int i = 0; i < self->size; i++) 
    {
        a->grad[i] += grad[i] / b->data[i];
        b->grad[i] -= grad[i] * a->data[i] / (b->data[i] * b->data[i]);
    }

    if (a->backward) 
    {
        a->backward(a, a->grad);
    }
    if (b->backward) 
    {
        b->backward(b, b->grad);
    }
}

void tensor_matmul_backward(Tensor *self, float *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int n = b->shape[b->ndim - 1];

    int batch_size = self->size / (m * n);

    // Ensure gradients are allocated
    if (a->grad == NULL) {
        a->grad = (float*)calloc(a->size, sizeof(float));
    }
    if (b->grad == NULL) {
        b->grad = (float*)calloc(b->size, sizeof(float));
    }

    // Compute gradient w.r.t. a
    for (int batch = 0; batch < batch_size; batch++) 
    {
        for (int i = 0; i < m; i++) 
        {
            for (int l = 0; l < k; l++) 
            {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) 
                {
                    sum += grad[batch * m * n + i * n + j] * b->data[batch * k * n + l * n + j];
                }
                a->grad[batch * m * k + i * k + l] += sum;
            }
        }
    }

    // Compute gradient w.r.t. b
    for (int batch = 0; batch < batch_size; batch++) 
    {
        for (int l = 0; l < k; l++) 
        {
            for (int j = 0; j < n; j++) 
            {
                float sum = 0.0f;
                for (int i = 0; i < m; i++) 
                {
                    sum += grad[batch * m * n + i * n + j] * a->data[batch * m * k + i * k + l];
                }
                b->grad[batch * k * n + l * n + j] += sum;
            }
        }
    }

    if (a->backward) 
    {
        a->backward(a, a->grad);
    }
    if (b->backward) 
    {
        b->backward(b, b->grad);
    }
}

void tensor_reshape_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += grad[i];
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_transpose_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;
    int ndim = self->ndim;
    int *reverse_axes = (int *)malloc(ndim * sizeof(int));

    // Reverse the axes permutation stored in self->axes
    for (int i = 0; i < ndim; i++) 
    {
        reverse_axes[self->axes[i]] = i;
    }

    // Accumulate gradients for the original tensor based on the reverse transpose
    for (int i = 0; i < self->size; i++) 
    {
        int old_index = 0;
        int new_index = i;

        for (int j = 0; j < ndim; j++) 
        {
            int axis = reverse_axes[j];
            int coord = new_index / self->stride[j];
            new_index %= self->stride[j];
            old_index += coord * tensor->stride[axis];
        }
        tensor->grad[old_index] += grad[i];
    }

    free(reverse_axes);

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_max_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    // Determine the axis along which max was computed by comparing tensor and self shapes
    int axis = -1;
    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (j < self->ndim && tensor->shape[i] == self->shape[j]) 
        {
            j++;
        } 
        else 
        {
            axis = i;
            break;
        }
    }

    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (d == axis)
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * self->stride[k--];
        }

        if (tensor->data[i] == self->data[result_index]) 
        {
            tensor->grad[i] += grad[result_index];
        }
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_min_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    // Determine the axis along which min was computed by comparing tensor and self shapes
    int axis = -1;
    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (j < self->ndim && tensor->shape[i] == self->shape[j]) 
        {
            j++;
        } 
        else 
        {
            axis = i;
            break;
        }
    }

    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (d == axis)
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * self->stride[k--];
        }

        if (tensor->data[i] == self->data[result_index]) 
        {
            tensor->grad[i] += grad[result_index];
        }
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_sum_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        // Calculate the axis dynamically
        int axis = -1;
        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (k < 0 || tensor->stride[d] != self->stride[k]) 
            {
                axis = d;
                break;
            }
            k--;
        }

        // Compute the index for the result tensor excluding the axis dimension
        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (d == axis) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * self->stride[k--];
        }

        tensor->grad[i] += grad[result_index];
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_mean_backward(Tensor *self, float *grad) 
{
    Tensor *tensor = self->grad_a;

    int axis = -1;
    for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
    {
        if (k < 0 || tensor->stride[d] != self->stride[k]) 
        {
            axis = d;
            break;
        }
        k--;
    }

    // Calculate the divisor
    int divisor = tensor->shape[axis];

    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        // Compute the index for the result tensor excluding the axis dimension
        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (d == axis) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * self->stride[k--];
        }

        tensor->grad[i] += grad[result_index] / divisor;
    }

    if (tensor->backward) 
    {
        tensor->backward(tensor, tensor->grad);
    }
}

void tensor_cat_backward(Tensor *self, float *grad) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    // Infer the axis by comparing the shapes of a, b, and the concatenated tensor
    int axis = -1;
    for (int i = 0; i < a->ndim; i++) 
    {
        if (a->shape[i] != b->shape[i]) 
        {
            axis = i;
            break;
        }
    }

    // Distribute the gradient to the first tensor
    for (int i = 0; i < a->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = a->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            int coord = (old_index / a->stride[d]) % a->shape[d];
            result_index += coord * self->stride[k--];
        }

        a->grad[i] += grad[result_index];
    }

    // Distribute the gradient to the second tensor
    for (int i = 0; i < b->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = b->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            int coord = (old_index / b->stride[d]) % b->shape[d];
            if (d == axis) 
            {
                coord += a->shape[axis];
            }
            result_index += coord * self->stride[k--];
        }

        b->grad[i] += grad[result_index];
    }

    if (a->backward) 
    {
        a->backward(a, a->grad);
    }
    if (b->backward) 
    {
        b->backward(b, b->grad);
    }
}