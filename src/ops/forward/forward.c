#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"

Tensor* tensor_negate(Tensor *tensor) 
{
    Tensor *result = allocate_tensor_with_same_shape(tensor);

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
    Tensor *result = allocate_tensor_with_same_shape(tensor);

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
    Tensor *result = allocate_tensor_with_same_shape(tensor);

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
    Tensor *result = allocate_tensor_with_same_shape(tensor);

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
    Tensor *result = allocate_tensor_with_same_shape(tensor);

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
    if (!check_shape_compatibility(a, b)) 
    {
        fprintf(stderr, "Error: Shape mismatch in tensor_add.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = allocate_tensor_with_same_shape(a);

    if (a == b) 
    {
        for (int i = 0; i < a->size; i++) 
        {
            result->data[i] = 2 * a->data[i];
        }
    } 
    else 
    {
        for (int i = 0; i < a->size; i++) 
        {
            result->data[i] = a->data[i] + b->data[i];
        }
    }

    result->backward = &tensor_add_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_sub(Tensor *a, Tensor *b) 
{
    if (!check_shape_compatibility(a, b)) 
    {
        fprintf(stderr, "Error: Shape mismatch in tensor_sub.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = allocate_tensor_with_same_shape(a);

    if (a == b) 
    {
        for (int i = 0; i < a->size; i++) 
        {
            result->data[i] = 0;
        }
    } 
    else 
    {
        for (int i = 0; i < a->size; i++) 
        {
            result->data[i] = a->data[i] - b->data[i];
        }
    }

    result->backward = &tensor_sub_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_mul(Tensor *a, Tensor *b) 
{
    if (!check_shape_compatibility(a, b)) 
    {
        fprintf(stderr, "Error: Shape mismatch in tensor_mul.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = allocate_tensor_with_same_shape(a);

    if (a == b) 
    {
        for (int i = 0; i < a->size; i++) 
        {
            result->data[i] = a->data[i] * a->data[i];
        }
    } 
    else 
    {
        for (int i = 0; i < a->size; i++) 
        {
            result->data[i] = a->data[i] * b->data[i];
        }
    }

    result->backward = &tensor_mul_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_div(Tensor *a, Tensor *b) 
{
    if (!check_shape_compatibility(a, b)) 
    {
        fprintf(stderr, "Error: Shape mismatch in tensor_div.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = allocate_tensor_with_same_shape(a);

    if (a == b) 
    {
        for (int i = 0; i < a->size; i++) 
        {
            result->data[i] = 1.0;
        }
    } 
    else 
    {
        for (int i = 0; i < a->size; i++) 
        {
            result->data[i] = a->data[i] / b->data[i];
        }
    }

    result->backward = &tensor_div_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}

Tensor* tensor_matmul(Tensor *a, Tensor *b) 
{
    int a_ndim = a->ndim;
    int b_ndim = b->ndim;

    // Handle cases where dimensions are less than 2
    if (a_ndim == 1) 
    {
        a = tensor_reshape(a, (int[]){1, a->size}, 2);
        a_ndim = 2;  // Update ndim after reshaping
    }
    if (b_ndim == 1) 
    {
        b = tensor_reshape(b, (int[]){b->size, 1}, 2);
        b_ndim = 2;  // Update ndim after reshaping
    }

    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int n = b->shape[b->ndim - 1];

    // Determine the output shape and size
    int out_ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;
    int *out_shape = (int*)malloc(out_ndim * sizeof(int));
    int *out_stride = (int*)malloc(out_ndim * sizeof(int));

    for (int i = 0; i < out_ndim - 2; i++) 
    {
        out_shape[i] = (a->ndim >= b->ndim) ? a->shape[i] : b->shape[i];
    }
    out_shape[out_ndim - 2] = m;
    out_shape[out_ndim - 1] = n;

    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->shape = out_shape;
    result->stride = out_stride;
    result->ndim = out_ndim;
    result->size = 1;

    // Calculate the strides for the result tensor
    result->stride[out_ndim - 1] = 1;
    for (int i = out_ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

    for (int i = 0; i < out_ndim; i++) 
    {
        result->size *= out_shape[i];
    }
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)calloc(result->size, sizeof(float));

    int batch_size = result->size / (m * n);

    for (int batch = 0; batch < batch_size; batch++) 
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

    return result;
}

Tensor* tensor_reshape(Tensor *tensor, int *new_shape, int new_ndim) 
{
    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    
    result->ndim = new_ndim;
    result->shape = (int*)malloc(new_ndim * sizeof(int));
    result->stride = (int*)malloc(new_ndim * sizeof(int));

    result->size = 1;
    for (int i = 0; i < new_ndim; i++) 
    {
        result->shape[i] = new_shape[i];
        result->size *= new_shape[i];
    }

    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)malloc(result->size * sizeof(float));
    memcpy(result->data, tensor->data, tensor->size * sizeof(float));
    memcpy(result->grad, tensor->grad, tensor->size * sizeof(float));

    result->stride[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

    result->backward = &tensor_reshape_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_transpose(Tensor *tensor, int *axes) 
{
    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->ndim = tensor->ndim;
    result->shape = (int*)malloc(result->ndim * sizeof(int));
    result->stride = (int*)malloc(result->ndim * sizeof(int));
    result->axes = (int*)malloc(result->ndim * sizeof(int));
    result->size = tensor->size;
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)calloc(result->size, sizeof(float));

    for (int i = 0; i < result->ndim; i++) 
    {
        result->shape[i] = tensor->shape[axes[i]];
        result->axes[i] = axes[i];
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
    int new_ndim = tensor->ndim - 1;
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->ndim = new_ndim;
    result->shape = new_shape;
    result->stride = (int*)malloc(new_ndim * sizeof(int));
    result->size = 1;
    for (int i = 0; i < new_ndim; i++) 
    {
        result->size *= result->shape[i];
    }
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)calloc(result->size, sizeof(float));

    result->stride[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

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

    result->backward = &tensor_max_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_min(Tensor *tensor, int axis) 
{
    int new_ndim = tensor->ndim - 1;
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->ndim = new_ndim;
    result->shape = new_shape;
    result->stride = (int*)malloc(new_ndim * sizeof(int));
    result->size = 1;
    for (int i = 0; i < new_ndim; i++) 
    {
        result->size *= result->shape[i];
    }
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)calloc(result->size, sizeof(float));

    result->stride[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

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

    result->backward = &tensor_min_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_argmax(Tensor *tensor, int axis) 
{
    int new_ndim = tensor->ndim - 1;
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    // Determine the new shape after removing the axis dimension
    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->ndim = new_ndim;
    result->shape = new_shape;
    result->stride = (int*)malloc(new_ndim * sizeof(int));
    result->size = 1;
    for (int i = 0; i < new_ndim; i++) 
    {
        result->size *= result->shape[i];
    }
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = NULL; // No gradient for argmax

    result->stride[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

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

        // Update the result data with the index where the max value is found
        if (result->data[result_index] == -1 || tensor->data[i] > tensor->data[result_index * tensor->shape[axis] + old_index % tensor->stride[axis]]) 
        {
            result->data[result_index] = old_index / tensor->stride[axis] % tensor->shape[axis];
        }
    }

    return result;
}

Tensor* tensor_argmin(Tensor *tensor, int axis) 
{
    int new_ndim = tensor->ndim - 1;
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    // Determine the new shape after removing the axis dimension
    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->ndim = new_ndim;
    result->shape = new_shape;
    result->stride = (int*)malloc(new_ndim * sizeof(int));
    result->size = 1;
    for (int i = 0; i < new_ndim; i++) 
    {
        result->size *= result->shape[i];
    }
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = NULL; // No gradient for argmin

    result->stride[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

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

        // Update the result data with the index where the min value is found
        if (result->data[result_index] == -1 || tensor->data[i] < tensor->data[result_index * tensor->shape[axis] + old_index % tensor->stride[axis]]) 
        {
            result->data[result_index] = old_index / tensor->stride[axis] % tensor->shape[axis];
        }
    }

    return result;
}

Tensor* tensor_sum(Tensor *tensor, int axis) 
{
    int new_ndim = tensor->ndim - 1;
    int *new_shape = (int*)malloc(new_ndim * sizeof(int));

    for (int i = 0, j = 0; i < tensor->ndim; i++) 
    {
        if (i != axis) 
        {
            new_shape[j++] = tensor->shape[i];
        }
    }

    Tensor *result = (Tensor*)malloc(sizeof(Tensor));
    result->ndim = new_ndim;
    result->shape = new_shape;
    result->stride = (int*)malloc(new_ndim * sizeof(int));
    result->size = 1;
    for (int i = 0; i < new_ndim; i++) 
    {
        result->size *= result->shape[i];
    }
    result->data = (float*)malloc(result->size * sizeof(float));
    result->grad = (float*)calloc(result->size, sizeof(float));

    result->stride[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

    // Initialize the result tensor data with zeros
    memset(result->data, 0, result->size * sizeof(float));

    // Perform the sum operation
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

        result->data[result_index] += tensor->data[i];
    }

    result->backward = &tensor_sum_backward;
    result->grad_a = tensor;

    return result;
}

Tensor* tensor_mean(Tensor *tensor, int axis) 
{
    Tensor *sum_result = tensor_sum(tensor, axis);
    
    int divisor = tensor->shape[axis];
    for (int i = 0; i < sum_result->size; i++) 
    {
        sum_result->data[i] /= divisor;
    }

    sum_result->backward = &tensor_mean_backward;
    sum_result->grad_a = tensor;

    return sum_result;
}

Tensor* tensor_cat(Tensor *a, Tensor *b, int axis) 
{
    // Check if the tensors can be concatenated along the given axis
    for (int i = 0; i < a->ndim; i++) 
    {
        if (i != axis && a->shape[i] != b->shape[i]) 
        {
            fprintf(stderr, "Error: Tensors cannot be concatenated along axis due to shape mismatch.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Calculate the shape of the result tensor
    int *new_shape = (int *)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) 
    {
        new_shape[i] = (i == axis) ? (a->shape[i] + b->shape[i]) : a->shape[i];
    }

    Tensor *result = (Tensor *)malloc(sizeof(Tensor));
    result->ndim = a->ndim;
    result->shape = new_shape;
    result->stride = (int *)malloc(result->ndim * sizeof(int));

    result->size = 1;
    for (int i = 0; i < result->ndim; i++) 
    {
        result->size *= result->shape[i];
    }
    result->data = (float *)malloc(result->size * sizeof(float));
    result->grad = (float *)calloc(result->size, sizeof(float));

    result->stride[result->ndim - 1] = 1;
    for (int i = result->ndim - 2; i >= 0; i--) 
    {
        result->stride[i] = result->stride[i + 1] * result->shape[i + 1];
    }

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

    result->backward = &tensor_cat_backward;
    result->grad_a = a;
    result->grad_b = b;

    return result;
}