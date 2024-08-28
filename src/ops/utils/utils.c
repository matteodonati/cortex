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
    for (int dim = 0; dim < (a->ndim > b->ndim ? a->ndim : b->ndim); dim++) 
    {
        int a_dim = a->ndim - 1 - dim;
        int b_dim = b->ndim - 1 - dim;

        int a_shape = (a_dim >= 0) ? a->shape[a_dim] : 1;
        int b_shape = (b_dim >= 0) ? b->shape[b_dim] : 1;

        int stride = index % (a_shape > b_shape ? a_shape : b_shape);
        index /= (a_shape > b_shape ? a_shape : b_shape);

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