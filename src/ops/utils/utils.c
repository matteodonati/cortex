#include <stdlib.h>
#include <string.h>
#include "ops/utils/utils.h"

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