#include <stdio.h>
#include <string.h>
#include "utils/tensor/tensor.h"

static void print_array_recursive(const float *array, size_t ndim, const size_t *shape, const size_t *stride, size_t current_dim, size_t *indices) 
{
    if (current_dim >= ndim) 
    {
        return;
    }

    printf("[");
    for (size_t i = 0; i < shape[current_dim]; i++) 
    {
        indices[current_dim] = i;

        if (current_dim == ndim - 1) 
        {
            size_t flat_index = 0;
            for (size_t j = 0; j < ndim; j++)
            {
                flat_index += indices[j] * stride[j];
            }

            float value = array[flat_index];
            printf("% .2e", value);
        }
        else 
        {
            print_array_recursive(array, ndim, shape, stride, current_dim + 1, indices);
        }

        if (i < shape[current_dim] - 1) 
        {
            printf(", ");
            if (current_dim < ndim - 1)
            {
                printf("\n");
                for (size_t j = 0; j <= current_dim; j++)
                {
                    printf(" ");
                }
            }
        }
    }
    printf("]");
}

void print_tensor(const tensor_t *tensor, const char *name) 
{
    if (tensor == NULL)
    {
        printf("Tensor is NULL\n");
        return;
    }

    printf("%s\n", name);
    printf("shape: (");
    for (size_t i = 0; i < tensor->ndim; i++) 
    {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->ndim - 1) 
        {
            printf(", ");
        }
    }
    printf(")\n");

    size_t indices[MAX_DIMS];
    
    printf("data:\n");
    if (tensor->data != NULL)
    {
        memset(indices, 0, sizeof(indices));
        print_array_recursive(tensor->data, tensor->ndim, tensor->shape, tensor->stride, 0, indices);
    }
    else
    {
        printf("None");
    }
    printf("\ngrad:\n");
    if (tensor->grad != NULL)
    {
        memset(indices, 0, sizeof(indices));
        print_array_recursive(tensor->grad, tensor->ndim, tensor->shape, tensor->stride, 0, indices);
    }
    else
    {
        printf("None");
    }
    printf("\n");
}