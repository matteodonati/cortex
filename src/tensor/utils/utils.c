#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor/tensor.h"
#include "tensor/utils/utils.h"

void tensor_set_slice(Tensor *dest, Tensor *src, int slice_index) 
{
    int slice_size = src->size;
    int offset = slice_index * slice_size;
    memcpy(&dest->data[offset], src->data, slice_size * sizeof(float));
}

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

void print_tensor(Tensor *tensor, const char *name) 
{
    printf("Tensor %s:\n", name);
    printf("shape:\n");
    print_tensor_shape(tensor);
    printf("data:\n");
    print_tensor_data(tensor);
    printf("grad:\n");
    print_tensor_grad(tensor);
    printf("\n");
}