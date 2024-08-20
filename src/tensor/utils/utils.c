#include <stdio.h>
#include <stdlib.h>
#include "tensor/tensor.h"
#include "tensor/utils/utils.h"

void print_tensor_shape(Tensor *tensor)
{
    printf("(");
    for(int i = 0; i < tensor->ndim; i++) 
    {
        printf("%d", tensor->shape[i]);
        if(i < tensor->ndim - 1) 
        {
            printf(", ");
        }
    }
    printf(")\n");
}

void print_tensor_array_recursive(float *array, int *shape, int ndim, int *indices, int level) 
{
    if (level == ndim - 1) 
    {
        // Print the innermost dimension
        printf("[");
        for (int i = 0; i < shape[level]; i++) 
        {
            if (i < PRINT_LIMIT || i >= shape[level] - PRINT_LIMIT) 
            {
                printf("%f", array[indices[level]]);
                if (i < shape[level] - 1) 
                {
                    printf(", ");
                }
            } 
            else if (i == PRINT_LIMIT) 
            {
                printf("..., ");
            }
            indices[level]++;
        }
        printf("]");
    } 
    else 
    {
        // Print outer dimensions
        printf("[");
        int rows_to_print = (shape[level] <= 2 * PRINT_LIMIT) ? shape[level] : PRINT_LIMIT;

        // Print the first part
        for (int i = 0; i < rows_to_print; i++) 
        {
            if (i > 0) 
            {
                printf("\n%*s", (level + 1), ""); // Indentation for nested levels
            }
            print_tensor_array_recursive(array, shape, ndim, indices, level + 1);
        }

        // Handle ellipsis if needed
        if (shape[level] > 2 * PRINT_LIMIT) 
        {
            printf("\n%*s...", (level + 1), "");  // Indentation for ellipsis
            for (int i = shape[level] - PRINT_LIMIT; i < shape[level]; i++) 
            {
                printf("\n%*s", (level + 1), "");  // Correct indentation for rows after ellipsis
                print_tensor_array_recursive(array, shape, ndim, indices, level + 1);
            }
        }
        printf("]");
    }
}

void print_tensor_data(Tensor *tensor) 
{
    int *indices = (int *)calloc(tensor->ndim, sizeof(int));
    if (tensor->data)
    {
        print_tensor_array_recursive(tensor->data, tensor->shape, tensor->ndim, indices, 0);
    }
    else
    {
        printf("None");
    }
    printf("\n");
    free(indices);
}

void print_tensor_grad(Tensor *tensor) 
{
    int *indices = (int *)calloc(tensor->ndim, sizeof(int));
    if (tensor->grad)
    {
        print_tensor_array_recursive(tensor->grad, tensor->shape, tensor->ndim, indices, 0);
    }
    else
    {
        printf("None");
    }
    printf("\n");
    free(indices);
}

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