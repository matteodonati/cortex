#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops/utils/utils.h"
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"

Tensor* tensor_negate(Tensor *tensor) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_negate.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = -tensor->data[i];
    }
    result->backward = &tensor_negate_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_abs(Tensor *tensor) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_abs.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = fabs(tensor->data[i]);
    }
    result->backward = &tensor_abs_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_sqrt(Tensor *tensor) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_sqrt.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = sqrtf(tensor->data[i]);
    }
    result->backward = &tensor_sqrt_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_exp(Tensor *tensor) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_exp.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = expf(tensor->data[i]);
    }
    result->backward = &tensor_exp_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_log(Tensor *tensor)
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_log.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++)
    {
        result->data[i] = logf(tensor->data[i]);
    }
    result->backward = &tensor_log_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_add(Tensor *a, Tensor *b) 
{
    if (a == NULL || b == NULL) 
    {
        fprintf(stderr, "Error: One of the input tensors is NULL in tensor_add.\n");
        exit(EXIT_FAILURE);
    }
    if (!check_shape_compatibility(a, b)) 
    {
        fprintf(stderr, "Error: Tensors are not broadcast-compatible in tensor_add.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, a->size >= b->size ? a : b);
    for (int i = 0; i < result->size; i++) 
    {
        int a_index, b_index;
        adjust_indices_for_broadcasting(a, b, &a_index, &b_index, i);
        result->data[i] = a->data[a_index] + b->data[b_index];
    }
    result->backward = &tensor_add_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_sub(Tensor *a, Tensor *b) 
{
    if (a == NULL || b == NULL) 
    {
        fprintf(stderr, "Error: One of the input tensors is NULL in tensor_sub.\n");
        exit(EXIT_FAILURE);
    }
    if (!check_shape_compatibility(a, b)) 
    {
        fprintf(stderr, "Error: Tensors are not broadcast-compatible in tensor_sub.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, a->size >= b->size ? a : b);
    for (int i = 0; i < result->size; i++) 
    {
        int a_index, b_index;
        adjust_indices_for_broadcasting(a, b, &a_index, &b_index, i);
        result->data[i] = a->data[a_index] - b->data[b_index];
    }
    result->backward = &tensor_sub_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_mul(Tensor *a, Tensor *b) 
{
    if (a == NULL || b == NULL) 
    {
        fprintf(stderr, "Error: One of the input tensors is NULL in tensor_mul.\n");
        exit(EXIT_FAILURE);
    }
    if (!check_shape_compatibility(a, b)) 
    {
        fprintf(stderr, "Error: Tensors are not broadcast-compatible in tensor_mul.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, a->size >= b->size ? a : b);
    for (int i = 0; i < result->size; i++) 
    {
        int a_index, b_index;
        adjust_indices_for_broadcasting(a, b, &a_index, &b_index, i);
        result->data[i] = a->data[a_index] * b->data[b_index];
    }
    result->backward = &tensor_mul_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_div(Tensor *a, Tensor *b) 
{
    if (a == NULL || b == NULL) 
    {
        fprintf(stderr, "Error: One of the input tensors is NULL in tensor_div.\n");
        exit(EXIT_FAILURE);
    }
    if (!check_shape_compatibility(a, b)) 
    {
        fprintf(stderr, "Error: Tensors are not broadcast-compatible in tensor_div.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, a->size >= b->size ? a : b);
    for (int i = 0; i < result->size; i++) 
    {
        int a_index, b_index;
        adjust_indices_for_broadcasting(a, b, &a_index, &b_index, i);
        result->data[i] = a->data[a_index] / b->data[b_index];
    }
    result->backward = &tensor_div_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_scalar_add(Tensor *tensor, float scalar) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_scalar_add.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = tensor->data[i] + scalar;
    }
    result->backward = &tensor_scalar_add_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_scalar_mul(Tensor *tensor, float scalar) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_scalar_mul.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_like(NULL, tensor);
    for (int i = 0; i < tensor->size; i++) 
    {
        result->data[i] = tensor->data[i] * scalar;
    }
    result->ops_utils.cached_float = scalar;
    result->backward = &tensor_scalar_mul_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_matmul(Tensor *a, Tensor *b) 
{
    if (a == NULL || b == NULL) 
    {
        fprintf(stderr, "Error: One of the input tensors is NULL in tensor_matmul.\n");
        exit(EXIT_FAILURE);
    }

    omp_set_num_threads(omp_get_num_procs());

    int a_ndim = a->ndim;
    int b_ndim = b->ndim;

    Tensor *result = NULL;

    // Case 1: 1D x 1D -> Dot product
    if (a_ndim == 1 && b_ndim == 1) 
    {
        if (a->size != b->size) 
        {
            fprintf(stderr, "Error: Incompatible dimensions for dot product.\n");
            exit(EXIT_FAILURE);
        }

        // Initialize result
        result = tensor_zeros(NULL, (int[]){1}, 1);

        float sum = 0.0f;
        for (int i = 0; i < a->size; i++) 
        {
            sum += a->data[i] * b->data[i];
        }
        result->data[0] = sum;
    }

    // Case 2: 1D x 2D -> Treat the 1D tensor as a row vector, then remove the added dimension
    else if (a_ndim == 1 && b_ndim == 2) 
    {
        int k = a->shape[0];  // a's original size is k
        int n = b->shape[1];  // b's second dimension is n
        int k_b = b->shape[0];  // b's first dimension should match a's

        if (k != k_b) 
        {
            fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication (1D x 2D).\n");
            exit(EXIT_FAILURE);
        }

        // Initialize result: (1, n) will be flattened to (n)
        result = tensor_zeros(NULL, (int[]){n}, 1);

        for (int j = 0; j < n; j++) 
        {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) 
            {
                sum += a->data[l] * b->data[l * b->stride[0] + j * b->stride[1]];
            }
            result->data[j] = sum;
        }
    }

    // Case 3: 2D x 1D -> Treat the 1D tensor as a column vector and return 1D result
    else if (a_ndim == 2 && b_ndim == 1) 
    {
        int m = a->shape[0];  // a's first dimension
        int k = a->shape[1];  // a's second dimension
        int k_b = b->shape[0];  // b's only dimension should match a's second dimension

        if (k != k_b) 
        {
            fprintf(stderr, "Error: Incompatible dimensions for matrix-vector multiplication.\n");
            exit(EXIT_FAILURE);
        }

        // Initialize result: (m)
        result = tensor_zeros(NULL, (int[]){m}, 1);

        for (int i = 0; i < m; i++) 
        {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) 
            {
                sum += a->data[i * a->stride[0] + l * a->stride[1]] * b->data[l];
            }
            result->data[i] = sum;
        }
    }

    // Case 4: 2D x 2D -> Matrix multiplication
    if (a_ndim == 2 && b_ndim == 2) 
    {
        int m = a->shape[0];
        int k = a->shape[1];
        int k_b = b->shape[0];
        int n = b->shape[1];

        if (k != k_b) 
        {
            fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication (2D x 2D).\n");
            exit(EXIT_FAILURE);
        }

        // Initialize result (m, n)
        result = tensor_zeros(NULL, (int[]){m, n}, 2);

        // Parallelize the loops with OpenMP
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) 
                {
                    sum += a->data[i * a->stride[0] + l * a->stride[1]] * b->data[l * b->stride[0] + j * b->stride[1]];
                }
                result->data[i * result->stride[0] + j * result->stride[1]] = sum;
            }
        }
    }

    // Case 5: Batched matrix multiplication (N-Dimensional inputs)
    else if (a_ndim > 2 || b_ndim > 2) 
    {
        int max_ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;
        int out_shape[max_ndim];

        // Align dimensions for broadcasting, handling batch dims
        for (int i = 0; i < max_ndim - 2; i++) 
        {
            int a_dim = (i < a_ndim - 2) ? a->shape[i] : 1;
            int b_dim = (i < b_ndim - 2) ? b->shape[i] : 1;

            if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) 
            {
                fprintf(stderr, "Error: Non-broadcastable batch dimensions in tensor_matmul.\n");
                exit(EXIT_FAILURE);
            }

            out_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
        }

        int m = a->shape[a_ndim - 2];
        int k_a = a->shape[a_ndim - 1];
        int k_b = b->shape[b_ndim - 2];
        int n = b->shape[b_ndim - 1];

        if (k_a != k_b) 
        {
            fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication.\n");
            exit(EXIT_FAILURE);
        }

        out_shape[max_ndim - 2] = m;
        out_shape[max_ndim - 1] = n;

        // Create the result tensor
        result = tensor_zeros(NULL, out_shape, max_ndim);

        // Perform batched matrix multiplication
        int num_batches = result->size / (m * n);

        // Parallelize over batches
        #pragma omp parallel for
        for (int batch = 0; batch < num_batches; batch++) 
        {
            for (int i = 0; i < m; i++) 
            {
                for (int j = 0; j < n; j++) 
                {
                    float sum = 0.0f;
                    for (int l = 0; l < k_a; l++) 
                    {
                        int a_batch_offset = (a_ndim > 2) ? batch * a->stride[0] : 0;
                        int b_batch_offset = (b_ndim > 2) ? batch * b->stride[0] : 0;

                        int a_index = a_batch_offset + i * a->stride[a_ndim - 2] + l * a->stride[a_ndim - 1];
                        int b_index = b_batch_offset + l * b->stride[b_ndim - 2] + j * b->stride[b_ndim - 1];

                        sum += a->data[a_index] * b->data[b_index];
                    }

                    int result_index = batch * result->stride[0] + i * result->stride[max_ndim - 2] + j * result->stride[max_ndim - 1];
                    result->data[result_index] = sum;
                }
            }
        }
    }

    else
    {
        fprintf(stderr, "Error: Unsupported input dimensions in tensor_matmul.\n");
        exit(EXIT_FAILURE);
    }

    result->backward = &tensor_matmul_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_reshape(Tensor *tensor, int *new_shape, int new_ndim) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_reshape.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    memcpy(result->data, tensor->data, tensor->size * sizeof(float));
    memcpy(result->grad, tensor->grad, tensor->size * sizeof(float));
    result->backward = &tensor_reshape_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_transpose(Tensor *tensor, int *axes) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_transpose.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_zeros(NULL, tensor->shape, tensor->ndim);
    result->ops_utils.cached_ints = (int *)malloc(result->ndim * sizeof(int));

    for (int i = 0; i < result->ndim; i++) 
    {
        result->shape[i] = tensor->shape[axes[i]];
        result->ops_utils.cached_ints[i] = axes[i];
    }

    result->stride[result->ndim - 1] = 1;
    for (int i = result->ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

    for (int i = 0; i < tensor->size; i++) 
    {
        int old_index = 0;
        int new_index = i;

        for (int j = 0; j < result->ndim; j++) {
            int axis = axes[j];
            int coord = new_index / result->stride[j];
            new_index %= result->stride[j];
            old_index += coord * tensor->stride[axis];
        }
        result->data[i] = tensor->data[old_index];
    }

    result->backward = &tensor_transpose_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_max(Tensor *tensor, int axis) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_max.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - 1;
    int new_shape[new_ndim];

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    tensor_extreme(tensor, result, axis, true);

    result->ops_utils.cached_int = axis;
    result->backward = &tensor_max_min_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_min(Tensor *tensor, int axis) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_min.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - 1;
    int new_shape[new_ndim];

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    tensor_extreme(tensor, result, axis, false);

    result->ops_utils.cached_int = axis;
    result->backward = &tensor_max_min_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_argmax(Tensor *tensor, int axis) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_argmax.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - 1;
    int new_shape[new_ndim];

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    tensor_arg_extreme(tensor, result, axis, true);

    return result;
}

Tensor* tensor_argmin(Tensor *tensor, int axis) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_argmin.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - 1;
    int new_shape[new_ndim];

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    tensor_arg_extreme(tensor, result, axis, false);

    return result;
}

Tensor* tensor_sum(Tensor *tensor, int *axes, int num_axes) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_sum.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - num_axes;
    int new_shape[new_ndim];

    // Determine reduce mask and divisor (divisor is not used for sum)
    int tmp;
    int reduce_mask[tensor->ndim];
    compute_reduce_mask_and_divisor(tensor, axes, num_axes, reduce_mask, &tmp);

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (!reduce_mask[i]) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);

    // Perform the summing operation
    tensor_reduce(tensor, result, reduce_mask);

    int *axes_copy = (int*)malloc(num_axes * sizeof(int));
    memcpy(axes_copy, axes, num_axes * sizeof(int));

    result->ops_utils.cached_ints = axes_copy;
    result->ops_utils.cached_int = num_axes;
    result->backward = &tensor_sum_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_mean(Tensor *tensor, int *axes, int num_axes) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_mean.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - num_axes;
    int new_shape[new_ndim];

    // Determine reduce mask and divisor
    int divisor;
    int reduce_mask[tensor->ndim];
    compute_reduce_mask_and_divisor(tensor, axes, num_axes, reduce_mask, &divisor);

    // Create the new shape for the result tensor
    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (!reduce_mask[i]) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);

    // Perform the summing operation
    tensor_reduce(tensor, result, reduce_mask);

    // Divide by the divisor
    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] /= divisor;
    }

    int *axes_copy = (int*)malloc(num_axes * sizeof(int));
    memcpy(axes_copy, axes, num_axes * sizeof(int));

    result->ops_utils.cached_ints = axes_copy;
    result->ops_utils.cached_int = num_axes;
    result->backward = &tensor_mean_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_var(Tensor *tensor, int *axes, int num_axes, bool unbiased) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_var.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - num_axes;
    int new_shape[new_ndim];

    // Determine reduce mask and divisor
    int divisor;
    int reduce_mask[tensor->ndim];
    compute_reduce_mask_and_divisor(tensor, axes, num_axes, reduce_mask, &divisor);

    // Adjust for unbiased variance if necessary
    int unbiased_divisor = unbiased ? divisor - 1 : divisor;

    // Create the new shape for the result tensor
    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (!reduce_mask[i]) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    Tensor *mean = tensor_zeros(NULL, new_shape, new_ndim);

    // Compute the mean
    tensor_reduce(tensor, mean, reduce_mask);
    for (int i = 0; i < result->size; i++) 
    {
        mean->data[i] /= divisor;
    }

    // Compute the variance
    for (int i = 0; i < tensor->size; i++) 
    {
        int var_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = new_ndim - 1; d >= 0; d--) 
        {
            if (reduce_mask[d]) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            var_index += coord * result->stride[k--];
        }
        float diff = tensor->data[i] - mean->data[var_index];
        result->data[var_index] += diff * diff;
    }

    // Free mean
    tensor_free(mean);

    // Divide by the unbiased divisor to compute variance
    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] /= unbiased_divisor;
    }

    // Cache axes for backward pass
    int *axes_copy = (int*)malloc(num_axes * sizeof(int));
    memcpy(axes_copy, axes, num_axes * sizeof(int));

    result->ops_utils.cached_ints = axes_copy;
    result->ops_utils.cached_int = num_axes;
    result->backward = &tensor_var_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_cat(Tensor *a, Tensor *b, int axis) 
{
    if (a == NULL || b == NULL) 
    {
        fprintf(stderr, "Error: One of the input tensors is NULL in tensor_cat.\n");
        exit(EXIT_FAILURE);
    }

    int new_shape[a->ndim];
    for (int i = 0; i < a->ndim; i++) 
    {
        new_shape[i] = (i == axis) ? (a->shape[i] + b->shape[i]) : a->shape[i];
    }

    Tensor *result = tensor_zeros(NULL, new_shape, a->ndim);

    // Copy data from the first tensor to the result tensor
    for (int i = 0; i < a->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = result->ndim - 1, k = result->ndim - 1; d >= 0; d--) 
        {
            int coord = (old_index / a->stride[d]) % a->shape[d];
            result_index += coord * result->stride[k--];
        }

        result->data[result_index] = a->data[i];
    }

    // Copy data from the second tensor to the result tensor
    for (int i = 0; i < b->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = result->ndim - 1, k = result->ndim - 1; d >= 0; d--) 
        {
            int coord = (old_index / b->stride[d]) % b->shape[d];
            if (d == axis) 
            {
                coord += a->shape[axis];
            }
            result_index += coord * result->stride[k--];
        }

        result->data[result_index] = b->data[i];
    }

    result->ops_utils.cached_int = axis;
    result->backward = &tensor_cat_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* im2col(Tensor *input, int kernel_height, int kernel_width, int stride_height, int stride_width, int pad_height, int pad_width)
{
    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int input_height = input->shape[2];
    int input_width = input->shape[3];

    // Calculate the output dimensions
    int output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

    // The dimensions of the resulting column matrix
    int col_height = in_channels * kernel_height * kernel_width;
    int col_width = output_height * output_width;

    // Allocate memory for the column matrix
    Tensor *col = tensor_zeros(NULL, (int[]){batch_size, col_height, col_width}, 3);

    // Populate the column matrix
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
                            int col_index = ((c * kernel_height + kh) * kernel_width + kw) * col_width + oh * output_width + ow;
                            if (h >= 0 && h < input_height && w >= 0 && w < input_width) 
                            {
                                col->data[n * col_height * col_width + col_index] = input->data[n * in_channels * input_height * input_width + c * input_height * input_width + h * input_width + w];
                            }
                        }
                    }
                }
            }
        }
    }

    col->backward = col2im;
    col->grad_a = input;

    // Store necessary parameters for backward pass
    col->ops_utils.cached_ints = (int *)malloc(6 * sizeof(int));
    col->ops_utils.cached_ints[0] = kernel_height;
    col->ops_utils.cached_ints[1] = kernel_width;
    col->ops_utils.cached_ints[2] = stride_height;
    col->ops_utils.cached_ints[3] = stride_width;
    col->ops_utils.cached_ints[4] = pad_height;
    col->ops_utils.cached_ints[5] = pad_width;

    return col;
}

Tensor* tensor_normalize2d(Tensor *x, bool is_training, Tensor *mean, Tensor *var, int *axes, int num_axes, float epsilon) 
{
    Tensor *y = tensor_zeros(NULL, x->shape, x->ndim);

    // Determine reduce mask and divisor
    int divisor;
    int reduce_mask[x->ndim];
    compute_reduce_mask_and_divisor(x, axes, num_axes, reduce_mask, &divisor);

    if (is_training) 
    {
        // Compute the mean
        tensor_reduce(x, mean, reduce_mask);
        for (int i = 0; i < mean->size; i++) 
        {
            mean->data[i] /= divisor;
        }

        // Compute the variance
        for (int i = 0; i < x->size; i++) 
        {
            int var_index = 0;
            int old_index = i;

            for (int d = x->ndim - 1, k = var->ndim - 1; d >= 0; d--) 
            {
                if (reduce_mask[d]) 
                {
                    continue;
                }
                int coord = (old_index / x->stride[d]) % x->shape[d];
                var_index += coord * var->stride[k--];
            }
            float diff = x->data[i] - mean->data[var_index];
            var->data[var_index] += diff * diff;
        }
        for (int i = 0; i < var->size; i++) 
        {
            var->data[i] /= divisor;
        }
    }

    // Normalize
    for (int i = 0; i < x->size; i++) 
    {
        int mean_index = 0;
        int old_index = i;

        // Calculate the mean and var index by skipping reduced axes
        for (int d = x->ndim - 1, k = mean->ndim - 1; d >= 0; d--) 
        {
            if (reduce_mask[d]) 
            {
                continue;
            }
            int coord = (old_index / x->stride[d]) % x->shape[d];
            mean_index += coord * mean->stride[k--];
        }

        // Normalize: (x - mean) / stddev
        float stddev = sqrtf(var->data[mean_index] + epsilon);
        y->data[i] = (x->data[i] - mean->data[mean_index]) / stddev;
    }

    // Cache axes for backward pass
    int *axes_copy = (int*)malloc(num_axes * sizeof(int));
    memcpy(axes_copy, axes, num_axes * sizeof(int));

    y->backward = tensor_normalize2d_backward;
    y->grad_a = x;
    y->ops_utils.cached_tensors = (Tensor **)malloc(2 * sizeof(Tensor *));
    y->ops_utils.cached_tensors[0] = mean;
    y->ops_utils.cached_tensors[1] = var;
    y->ops_utils.cached_ints = axes_copy;
    y->ops_utils.cached_float = epsilon;
    y->ops_utils.cached_int = num_axes;

    return y;
}