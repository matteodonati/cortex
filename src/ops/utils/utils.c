#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ops/utils/utils.h"

int check_shape_compatibility(Tensor *a, Tensor *b) 
{
    int ndim_a = a->ndim;
    int ndim_b = b->ndim;
    int min_ndim = ndim_a < ndim_b ? ndim_a : ndim_b;

    for (int i = 0; i < min_ndim; i++) 
    {
        int dim_a = a->shape[ndim_a - 1 - i];
        int dim_b = b->shape[ndim_b - 1 - i];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) 
        {
            return 0;
        }
    }
    return 1;
}

void adjust_indices_for_broadcasting(Tensor *a, Tensor *b, int *a_index, int *b_index, int i) 
{
    *a_index = 0;
    *b_index = 0;

    int index = i;
    int a_shape, b_shape;
    int max_dim = (a->ndim > b->ndim) ? a->ndim : b->ndim;

    for (int dim = 0; dim < max_dim; dim++) 
    {
        int a_dim = a->ndim - 1 - dim;
        int b_dim = b->ndim - 1 - dim;

        a_shape = (a_dim >= 0) ? a->shape[a_dim] : 1;
        b_shape = (b_dim >= 0) ? b->shape[b_dim] : 1;

        int larger_shape = (a_shape > b_shape) ? a_shape : b_shape;
        int stride = index % larger_shape;
        index /= larger_shape;

        if (a_shape > 1 && a_dim >= 0) 
        {
            *a_index += stride * a->stride[a_dim];
        }

        if (b_shape > 1 && b_dim >= 0) 
        {
            *b_index += stride * b->stride[b_dim];
        }
    }
}

void tensor_extreme(Tensor *tensor, Tensor *result, int axis, bool is_max) 
{
    int new_ndim = result->ndim;
    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] = is_max ? -INFINITY : INFINITY;
    }

    // Perform the max or min operation
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

        if (is_max) 
        {
            if (tensor->data[i] > result->data[result_index]) 
            {
                result->data[result_index] = tensor->data[i];
            }
        } 
        else 
        {
            if (tensor->data[i] < result->data[result_index]) 
            {
                result->data[result_index] = tensor->data[i];
            }
        }
    }
}

void tensor_arg_extreme(Tensor *tensor, Tensor *result, int axis, bool is_argmax) 
{
    int new_ndim = result->ndim;

    for (int i = 0; i < result->size; i++) 
    {
        result->data[i] = -1;
    }

    // Perform the argmax or argmin operation
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

        if (result->data[result_index] == -1 || 
            ( is_argmax && tensor->data[i] > tensor->data[(int)result->data[result_index] * tensor->stride[axis] + old_index % tensor->stride[axis]]) ||
            (!is_argmax && tensor->data[i] < tensor->data[(int)result->data[result_index] * tensor->stride[axis] + old_index % tensor->stride[axis]]))
        {
            result->data[result_index] = old_index / tensor->stride[axis] % tensor->shape[axis];
        }
    }
}

void compute_reduce_mask_and_divisor(Tensor *tensor, int *axes, int num_axes, int *reduce_mask, int *divisor)
{
    *divisor = 1;
    for (int i = 0; i < tensor->ndim; i++) 
    {
        reduce_mask[i] = 0;
    }

    for (int i = 0; i < num_axes; i++) 
    {
        reduce_mask[axes[i]] = 1;
        *divisor *= tensor->shape[axes[i]];
    }
}

void tensor_reduce(Tensor *tensor, Tensor *result, int *reduce_mask) 
{
    int new_ndim = result->ndim;

    // Perform the sum operation over the specified axes
    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = new_ndim - 1; d >= 0; d--) 
        {
            if (reduce_mask[d]) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * result->stride[k--];
        }

        result->data[result_index] += tensor->data[i];
    }
}

void accumulate_grad(Tensor *self, Tensor *tensor, int *reduce_mask, int divisor, bool is_var, float *mean, bool apply_var)
{
    for (int i = 0; i < tensor->size; i++) 
    {
        int result_index = 0;
        int old_index = i;

        for (int d = tensor->ndim - 1, k = self->ndim - 1; d >= 0; d--) 
        {
            if (reduce_mask[d]) 
            {
                continue;
            }
            int coord = (old_index / tensor->stride[d]) % tensor->shape[d];
            result_index += coord * self->stride[k--];
        }

        if (is_var && apply_var) 
        {
            float diff = tensor->data[i] - mean[result_index];
            tensor->grad[i] += (2.0 / divisor) * diff * self->grad[result_index];
        } 
        else if (divisor > 1) 
        {
            tensor->grad[i] += self->grad[result_index] / divisor;
        } else 
        {
            tensor->grad[i] += self->grad[result_index];
        }
    }
}

void backward(Tensor *tensor)
{
    if (tensor->backward)
    {
        tensor->backward(tensor);
    }
}

Tensor* one_hot_encode(Tensor *labels, int num_classes) 
{
    int *new_shape = (int *)malloc((labels->ndim + 1) * sizeof(int));
    memcpy(new_shape, labels->shape, labels->ndim * sizeof(int));
    new_shape[labels->ndim] = num_classes;

    Tensor *one_hot = tensor_zeros("one_hot_labels", new_shape, labels->ndim + 1);
    for (int i = 0; i < labels->size; i++) 
    {
        int class_index = (int)labels->data[i];
        one_hot->data[i * num_classes + class_index] = 1.0;
    }

    free(new_shape);
    return one_hot;
}