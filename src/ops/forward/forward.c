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
        result->data[i] = sqrt(tensor->data[i]);
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
        result->data[i] = exp(tensor->data[i]);
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
        result->data[i] = log(tensor->data[i]);
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

    int a_ndim = a->ndim;
    int b_ndim = b->ndim;

    // Handle cases where dimensions are less than 2
    if (a_ndim == 1) 
    {
        a = tensor_reshape(a, (int[]){1, a->size}, 2);
        a_ndim = 2;
    }
    if (b_ndim == 1) 
    {
        b = tensor_reshape(b, (int[]){b->size, 1}, 2);
        b_ndim = 2;
    }

    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int n = b->shape[b->ndim - 1];

    // Output ndim
    int out_ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;

    // Output shape
    int *out_shape = (int*)malloc(out_ndim * sizeof(int));
    for (int i = 0; i < out_ndim - 2; i++) 
    {
        out_shape[i] = (a->ndim >= b->ndim) ? a->shape[i] : b->shape[i];
    }
    out_shape[out_ndim - 2] = m;
    out_shape[out_ndim - 1] = n;

    // Create the result Tensor
    Tensor *result = tensor_zeros(NULL, out_shape, out_ndim);

    // Perform matrix multiplication
    for (int batch = 0; batch < result->size / (m * n); batch++) 
    {
        for (int i = 0; i < m; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) 
                {
                    int a_index = batch * m * k + i * k + l;
                    int b_index = batch * k * n + l * n + j;
                    sum += a->data[a_index] * b->data[b_index];
                }
                result->data[batch * m * n + i * n + j] = sum;
            }
        }
    }

    result->backward = &tensor_matmul_backward;
    result->grad_a = a;
    result->grad_b = b;

    free(out_shape);

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
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);

    // Initialize the result tensor data with negative infinity
    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] = -INFINITY;
    }

    // Perform the max operation
    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        // Compute the index for the result tensor excluding the axis dimension
        for (int d = tensor->ndim - 1, k = new_ndim - 1; d >= 0; d--) 
        {
            if (d == axis) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * result->stride[k--];
        }

        // Update the result data
        if (tensor->data[i] > result->data[result_index]) 
        {
            result->data[result_index] = tensor->data[i];
        }
    }

    result->ops_utils.cached_int = axis;
    result->backward = &tensor_max_backward;
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
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);

    // Initialize the result tensor data with positive infinity
    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] = INFINITY;
    }

    // Perform the min operation
    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        // Compute the index for the result tensor excluding the axis dimension
        for (int d = tensor->ndim - 1, k = new_ndim - 1; d >= 0; d--) 
        {
            if (d == axis)
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * result->stride[k--];
        }

        // Update the result data
        if (tensor->data[i] < result->data[result_index]) 
        {
            result->data[result_index] = tensor->data[i];
        }
    }

    result->ops_utils.cached_int = axis;
    result->backward = &tensor_min_backward;
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
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    result->grad = NULL; // No gradient for argmax

    // Initialize the result tensor data with -1 (invalid index)
    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] = -1;
    }

    // Perform the argmax operation
    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = new_ndim - 1; d >= 0; d--) 
        {
            if (d == axis) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * result->stride[k--];
        }

        if (result->data[result_index] == -1 || tensor->data[i] > tensor->data[(int)result->data[result_index] * tensor->stride[axis] + old_index % tensor->stride[axis]]) 
        {
            result->data[result_index] = old_index / tensor->stride[axis] % tensor->shape[axis];
        }
    }

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
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    result->grad = NULL; // No gradient for argmin

    // Initialize the result tensor data with -1 (invalid index)
    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] = -1;
    }

    // Perform the argmin operation
    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = new_ndim - 1; d >= 0; d--) 
        {
            if (d == axis) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * result->stride[k--];
        }

        if (result->data[result_index] == -1 || tensor->data[i] < tensor->data[(int)result->data[result_index] * tensor->stride[axis] + old_index % tensor->stride[axis]]) 
        {
            result->data[result_index] = old_index / tensor->stride[axis] % tensor->shape[axis];
        }
    }

    return result;
}

Tensor* tensor_sum(Tensor *tensor, int axis) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_sum.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - 1;
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);

    // Perform the sum operation
    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = new_ndim - 1; d >= 0; d--) 
        {
            if (d == axis) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * result->stride[k--];
        }

        result->data[result_index] += tensor->data[i];
    }

    result->ops_utils.cached_int = axis;
    result->backward = &tensor_sum_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_mean(Tensor *tensor, int axis) 
{
    if (tensor == NULL) 
    {
        fprintf(stderr, "Error: Input tensor is NULL in tensor_mean.\n");
        exit(EXIT_FAILURE);
    }

    int new_ndim = tensor->ndim - 1;
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = tensor_zeros(NULL, new_shape, new_ndim);
    
    int divisor = tensor->shape[axis];

    // Perform sum
    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = new_ndim - 1; d >= 0; d--) 
        {
            if (d == axis) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * result->stride[k--];
        }

        result->data[result_index] += tensor->data[i];
    }

    // Divide by the number of elements along the axis
    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] /= divisor;
    }

    result->ops_utils.cached_int = axis;
    result->backward = &tensor_mean_backward;
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

    int *new_shape = (int *)malloc(a->ndim * sizeof(int));
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