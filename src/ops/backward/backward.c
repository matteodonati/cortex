#include <math.h>
#include <stdlib.h>
#include "ops/utils/utils.h"
#include "ops/backward/backward.h"

void tensor_negate_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] -= self->grad[i];
    }

    backward(tensor);
}

void tensor_abs_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += (tensor->data[i] >= 0 ? 1.0f : -1.0f) * self->grad[i];
    }

    backward(tensor);
}

void tensor_sqrt_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += 0.5f / sqrt(tensor->data[i]) * self->grad[i];
    }

    backward(tensor);
}

void tensor_exp_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += exp(tensor->data[i]) * self->grad[i];
    }

    backward(tensor);
}

void tensor_log_backward(Tensor *self)
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++)
    {
        tensor->grad[i] += self->grad[i] / tensor->data[i];
    }

    backward(tensor);
}

void tensor_add_backward(Tensor *self) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for (int i = 0; i < self->size; i++) 
    {
        int a_index, b_index;
        adjust_indices_for_broadcasting(a, b, &a_index, &b_index, i);
        a->grad[a_index] += self->grad[i];
        b->grad[b_index] += self->grad[i];
    }

    backward(a);
    backward(b);
}

void tensor_sub_backward(Tensor *self) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for (int i = 0; i < self->size; i++) 
    {
        int a_index, b_index;
        adjust_indices_for_broadcasting(a, b, &a_index, &b_index, i);
        a->grad[a_index] += self->grad[i];
        b->grad[b_index] -= self->grad[i];
    }

    backward(a);
    backward(b);
}

void tensor_mul_backward(Tensor *self) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for (int i = 0; i < self->size; i++) 
    {
        int a_index, b_index;
        adjust_indices_for_broadcasting(a, b, &a_index, &b_index, i);
        a->grad[a_index] += self->grad[i] * b->data[b_index];
        b->grad[b_index] += self->grad[i] * a->data[a_index];
    }

    backward(a);
    backward(b);
}

void tensor_div_backward(Tensor *self) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    for (int i = 0; i < self->size; i++) 
    {
        int a_index, b_index;
        adjust_indices_for_broadcasting(a, b, &a_index, &b_index, i);
        a->grad[a_index] += self->grad[i] / b->data[b_index];
        b->grad[b_index] -= self->grad[i] * a->data[a_index] / (b->data[b_index] * b->data[b_index]);
    }

    backward(a);
    backward(b);
}

void tensor_scalar_mul_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->grad[i] += self->grad[i] * tensor->ops_utils.cached_scalar;
    }

    backward(tensor);
}

void tensor_matmul_backward(Tensor *self) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int n = b->shape[b->ndim - 1];

    int batch_size = self->size / (m * n);

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
                    sum += self->grad[batch * m * n + i * n + j] * b->data[batch * k * n + l * n + j];
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
                    sum += self->grad[batch * m * n + i * n + j] * a->data[batch * m * k + i * k + l];
                }
                b->grad[batch * k * n + l * n + j] += sum;
            }
        }
    }

    backward(a);
    backward(b);
}

void tensor_reshape_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += self->grad[i];
    }

    backward(tensor);
}

void tensor_transpose_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    int ndim = self->ndim;
    int *reverse_axes = (int *)malloc(ndim * sizeof(int));

    // Reverse the axes permutation stored in self->axes
    for (int i = 0; i < ndim; i++) 
    {
        reverse_axes[self->ops_utils.cached_axes[i]] = i;
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
        tensor->grad[old_index] += self->grad[i];
    }

    free(reverse_axes);

    backward(tensor);
}

void tensor_max_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (d == self->ops_utils.cached_axis)
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * self->stride[k--];
        }

        if (tensor->data[i] == self->data[result_index]) 
        {
            tensor->grad[i] += self->grad[result_index];
        }
    }

    backward(tensor);
}

void tensor_min_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (d == self->ops_utils.cached_axis)
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * self->stride[k--];
        }

        if (tensor->data[i] == self->data[result_index]) 
        {
            tensor->grad[i] += self->grad[result_index];
        }
    }

    backward(tensor);
}

void tensor_sum_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (d == self->ops_utils.cached_axis) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * self->stride[k--];
        }

        tensor->grad[i] += self->grad[result_index];
    }

    backward(tensor);
}

void tensor_mean_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    // Calculate the divisor
    int axis = self->ops_utils.cached_axis;
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

        tensor->grad[i] += self->grad[result_index] / divisor;
    }

    backward(tensor);
}

void tensor_cat_backward(Tensor *self) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

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

        a->grad[i] += self->grad[result_index];
    }

    // Distribute the gradient to the second tensor
    int axis = self->ops_utils.cached_axis;
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

        b->grad[i] += self->grad[result_index];
    }

    backward(a);  
    backward(b);
}