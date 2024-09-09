#include <math.h>
#include <stdio.h>
#include <string.h>
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
        tensor->grad[i] += 0.5f / sqrtf(tensor->data[i]) * self->grad[i];
    }

    backward(tensor);
}

void tensor_exp_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < self->size; i++) 
    {
        tensor->grad[i] += expf(tensor->data[i]) * self->grad[i];
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

void tensor_scalar_add_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    
    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->grad[i] += self->grad[i];
    }

    backward(tensor);
}

void tensor_scalar_mul_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < tensor->size; i++) 
    {
        tensor->grad[i] += self->grad[i] * tensor->ops_utils.cached_float;
    }

    backward(tensor);
}

void tensor_matmul_backward(Tensor *self) 
{
    Tensor *a = self->grad_a;
    Tensor *b = self->grad_b;

    int a_ndim = a->ndim;
    int b_ndim = b->ndim;
    int result_ndim = self->ndim;

    // Case 1: Backward for 1D x 1D -> Dot product
    if (a_ndim == 1 && b_ndim == 1) 
    {
        for (int i = 0; i < a->size; i++) 
        {
            a->grad[i] += self->grad[0] * b->data[i];  // Grad for a
            b->grad[i] += self->grad[0] * a->data[i];  // Grad for b
        }
    }

    // Case 2: Backward for 1D x 2D -> Matrix multiply 1D as row vector
    else if (a_ndim == 1 && b_ndim == 2) 
    {
        int k = a->shape[0];  // a's original size is k
        int n = b->shape[1];  // b's second dimension is n

        for (int l = 0; l < k; l++) 
        {
            for (int j = 0; j < n; j++) 
            {
                int b_index = l * b->stride[0] + j * b->stride[1];
                a->grad[l] += self->grad[j] * b->data[b_index];  // Grad for a
                b->grad[b_index] += self->grad[j] * a->data[l];  // Grad for b
            }
        }
    }

    // Case 3: Backward for 2D x 1D -> Matrix multiply 1D as column vector
    else if (a_ndim == 2 && b_ndim == 1) 
    {
        int m = a->shape[0];
        int k = a->shape[1];

        for (int i = 0; i < m; i++) 
        {
            for (int l = 0; l < k; l++) 
            {
                a->grad[i * a->stride[0] + l * a->stride[1]] += self->grad[i] * b->data[l];  // Grad for a
                b->grad[l] += self->grad[i] * a->data[i * a->stride[0] + l * a->stride[1]];  // Grad for b
            }
        }
    }

    // Case 4: Backward for 2D x 2D -> Matrix multiplication
    else if (a_ndim == 2 && b_ndim == 2) 
    {
        int m = a->shape[0];
        int k = a->shape[1];
        int n = b->shape[1];

        for (int i = 0; i < m; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                for (int l = 0; l < k; l++) 
                {
                    a->grad[i * a->stride[0] + l * a->stride[1]] += self->grad[i * self->stride[0] + j * self->stride[1]] * b->data[l * b->stride[0] + j * b->stride[1]];  // Grad for a
                    b->grad[l * b->stride[0] + j * b->stride[1]] += self->grad[i * self->stride[0] + j * self->stride[1]] * a->data[i * a->stride[0] + l * a->stride[1]];  // Grad for b
                }
            }
        }
    }

    // Case 5: Backward for batched matrix multiplication (N-Dimensional inputs)
    else if (a_ndim > 2 || b_ndim > 2) 
    {
        // Determine the matrix dimensions
        int m = a->shape[a_ndim - 2];  // Rows of a
        int k_a = a->shape[a_ndim - 1];  // Columns of a / Rows of b
        int k_b = b->shape[b_ndim - 2];  // Rows of b (must match k_a)
        int n = b->shape[b_ndim - 1];  // Columns of b

        if (k_a != k_b) 
        {
            fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication in backward pass.\n");
            exit(EXIT_FAILURE);
        }

        // Determine batch dimensions
        int batch_size_a = (a_ndim > 2) ? a->shape[0] : 1;
        int batch_size_b = (b_ndim > 2) ? b->shape[0] : 1;
        int max_batch_size = (batch_size_a > batch_size_b) ? batch_size_a : batch_size_b;  // Use the larger batch size

        // Perform batched matrix multiplication backward
        for (int batch = 0; batch < max_batch_size; batch++) 
        {
            // Determine the effective batch index for broadcasting
            int a_batch_offset = (batch_size_a > 1) ? batch * a->stride[0] : 0;
            int b_batch_offset = (batch_size_b > 1) ? batch * b->stride[0] : 0;

            for (int i = 0; i < m; i++) 
            {
                for (int j = 0; j < n; j++) 
                {
                    float grad_value = self->grad[batch * self->stride[0] + i * self->stride[result_ndim - 2] + j * self->stride[result_ndim - 1]];

                    for (int l = 0; l < k_a; l++) 
                    {
                        int a_index = a_batch_offset + i * a->stride[a_ndim - 2] + l * a->stride[a_ndim - 1];
                        int b_index = b_batch_offset + l * b->stride[b_ndim - 2] + j * b->stride[b_ndim - 1];

                        // Accumulate gradients for 'a' and 'b'
                        a->grad[a_index] += grad_value * b->data[b_index];  // Grad for a
                        b->grad[b_index] += grad_value * a->data[a_index];  // Grad for b
                    }
                }
            }
        }
    }

    else
    {
        fprintf(stderr, "Error: Unsupported input dimensions in tensor_matmul_backward.\n");
        exit(EXIT_FAILURE);
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
    int reverse_axes[ndim];

    // Reverse the axes permutation
    for (int i = 0; i < ndim; i++) 
    {
        reverse_axes[self->ops_utils.cached_ints[i]] = i;
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

    backward(tensor);
}

void tensor_max_min_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;

    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (d == self->ops_utils.cached_int)
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
    int *axes = self->ops_utils.cached_ints;
    int num_axes = self->ops_utils.cached_int;
    int reduce_mask[tensor->ndim];
    int divisor;

    // Compute the reduce_mask and accumulate gradients (no divisor needed for sum)
    compute_reduce_mask_and_divisor(tensor, axes, num_axes, reduce_mask, &divisor);
    accumulate_grad(self, tensor, reduce_mask, 1, false, NULL, false);

    backward(tensor);
}

void tensor_mean_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    int *axes = self->ops_utils.cached_ints;
    int num_axes = self->ops_utils.cached_int;
    int reduce_mask[tensor->ndim];
    int divisor;

    // Compute the reduce_mask and divisor, and accumulate gradient
    compute_reduce_mask_and_divisor(tensor, axes, num_axes, reduce_mask, &divisor);
    accumulate_grad(self, tensor, reduce_mask, divisor, false, NULL, false);

    backward(tensor);
}

void tensor_var_backward(Tensor *self) 
{
    Tensor *tensor = self->grad_a;
    int *axes = self->ops_utils.cached_ints;
    int num_axes = self->ops_utils.cached_int;
    int reduce_mask[tensor->ndim];
    float mean[self->size];
    int divisor;

    // Compute the reduce_mask and divisor
    compute_reduce_mask_and_divisor(tensor, axes, num_axes, reduce_mask, &divisor);

    // Unbiased variance divisor adjustment
    int unbiased = (divisor > 1) ? divisor - 1 : divisor;

    // Recompute the mean in the backward pass
    memset(mean, 0, self->size * sizeof(float));
    for (int i = 0; i < tensor->size; i++) 
    {
        int mean_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (reduce_mask[d]) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            mean_index += coord * self->stride[k--];
        }

        mean[mean_index] += tensor->data[i];
    }
    for (int i = 0; i < self->size; i++) 
    {
        mean[i] /= divisor;
    }

    // Accumulate gradients using variance formula
    accumulate_grad(self, tensor, reduce_mask, unbiased, true, mean, true);

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
    int axis = self->ops_utils.cached_int;
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

void col2im(Tensor *self) 
{
    Tensor *input = self->grad_a;

    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int input_height = input->shape[2];
    int input_width = input->shape[3];

    // Retrieve the parameters from the forward pass
    int kernel_height = self->ops_utils.cached_ints[0]; 
    int kernel_width = self->ops_utils.cached_ints[1];
    int stride_height = self->ops_utils.cached_ints[2];
    int stride_width = self->ops_utils.cached_ints[3];
    int pad_height = self->ops_utils.cached_ints[4];
    int pad_width = self->ops_utils.cached_ints[5];

    // Calculate the output dimensions
    int output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

    // Iterate over the batches and channels
    for (int n = 0; n < batch_size; n++) 
    {
        for (int c = 0; c < in_channels; c++) 
        {
            for (int kh = 0; kh < kernel_height; kh++) 
            {
                for (int kw = 0; kw < kernel_width; kw++) 
                {
                    for (int oh = 0; oh < output_height; oh++) 
                    {
                        for (int ow = 0; ow < output_width; ow++) 
                        {
                            int h = oh * stride_height - pad_height + kh;
                            int w = ow * stride_width - pad_width + kw;
                            int col_index = ((c * kernel_height + kh) * kernel_width + kw) * output_width * output_height + oh * output_width + ow;
                            if (h >= 0 && h < input_height && w >= 0 && w < input_width) 
                            {
                                input->grad[n * in_channels * input_height * input_width + c * input_height * input_width + h * input_width + w] += self->grad[n * self->shape[1] * self->shape[2] + col_index];
                            }
                        }
                    }
                }
            }
        }
    }

    backward(input);
}

void tensor_normalize2d_backward(Tensor *self) 
{
    Tensor *x = self->grad_a;
    Tensor *mean = self->ops_utils.cached_tensors[0]; // Cached mean
    Tensor *var = self->ops_utils.cached_tensors[1];  // Cached variance
    Tensor *dvar = tensor_zeros(NULL, var->shape, var->ndim);
    Tensor *dmean = tensor_zeros(NULL, mean->shape, mean->ndim);
    float epsilon = self->ops_utils.cached_float;
    int *axes = self->ops_utils.cached_ints;
    int num_axes = self->ops_utils.cached_int;

    // Recompute the divisor based on the axes
    int divisor;
    int reduce_mask[x->ndim];
    compute_reduce_mask_and_divisor(x, axes, num_axes, reduce_mask, &divisor);

    // Propagate gradients to mean and var
    for (int i = 0; i < x->size; i++) 
    {
        int mean_index = 0;
        int old_index = i;

        for (int d = x->ndim - 1, k = mean->ndim - 1; d >= 0; d--) 
        {
            if (reduce_mask[d]) 
            {
                continue;
            }
            int coord = (old_index / x->stride[d]) % x->shape[d];
            mean_index += coord * mean->stride[k--];
        }

        float diff = x->data[i] - mean->data[mean_index];
        dvar->data[mean_index] += self->grad[i] * diff * (-0.5f) * powf(var->data[mean_index] + epsilon, -1.5f);
        dmean->data[mean_index] += self->grad[i] * (-1.0f / sqrtf(var->data[mean_index] + epsilon));
    }
    for (int i = 0; i < dmean->size; i++) 
    {
        dmean->data[i] /= divisor;
    }

    // Propagate gradients to input x
    for (int i = 0; i < x->size; i++) 
    {
        int mean_index = 0;
        int old_index = i;

        for (int d = x->ndim - 1, k = mean->ndim - 1; d >= 0; d--) 
        {
            if (reduce_mask[d]) 
            {
                continue;
            }
            int coord = (old_index / x->stride[d]) % x->shape[d];
            mean_index += coord * mean->stride[k--];
        }

        float diff = x->data[i] - mean->data[mean_index];
        x->grad[i] += (self->grad[i] / sqrtf(var->data[mean_index] + epsilon)) + (dvar->data[mean_index] * 2 * diff / divisor) + dmean->data[mean_index];
    }

    tensor_free(dvar);
    tensor_free(dmean);

    backward(x);
}