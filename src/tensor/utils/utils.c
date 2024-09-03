#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor/tensor.h"
#include "tensor/utils/utils.h"

void print_tensor_array_recursive(float *data, int *shape, int ndim, int current_dim, int *indices) 
{
    if (current_dim == ndim - 1) 
    {
        printf("[");
        for (int i = 0; i < shape[current_dim]; i++) 
        {
            indices[current_dim] = i;
            int flat_index = 0;
            int stride = 1;
            for (int j = ndim - 1; j >= 0; j--)
            {
                flat_index += indices[j] * stride;
                stride *= shape[j];
            }
            printf("%.2e", data[flat_index]);  // Scientific notation with  digits after the decimal point
            if (i < shape[current_dim] - 1) 
            {
                printf(", ");
            }
        }
        printf("]");
    } 
    else 
    {
        printf("[");
        for (int i = 0; i < shape[current_dim]; i++) 
        {
            indices[current_dim] = i;
            print_tensor_array_recursive(data, shape, ndim, current_dim + 1, indices);
            if (i < shape[current_dim] - 1) 
            {
                printf(",\n");
                for (int j = 0; j <= current_dim; j++) 
                {
                    printf(" ");
                }
            }
        }
        printf("]");
    }
}

void print_tensor(Tensor *tensor, const char *name) 
{
    printf("Tensor %s\n", name);
    printf("shape: (");
    for (int i = 0; i < tensor->ndim; i++) 
    {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1) 
        {
            printf(", ");
        }
    }
    printf(")\n");

    int *indices = (int *)malloc(tensor->ndim * sizeof(int));
    printf("data:\n");
    if (tensor->data != NULL)
    {
        print_tensor_array_recursive(tensor->data, tensor->shape, tensor->ndim, 0, indices);
    }
    else
    {
        printf("None");
    }
    printf("\n");

    printf("grad:\n");
    if (tensor->grad != NULL) 
    {
        print_tensor_array_recursive(tensor->grad, tensor->shape, tensor->ndim, 0, indices);
    }
    else
    {
        printf("None");
    }
    printf("\n");

    free(indices);
}