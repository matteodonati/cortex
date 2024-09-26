#include <stdio.h>
#include "utils/tensor/tensor.h"

static void print_tensor_array_recursive(const tensor_t *tensor, size_t current_dim, size_t *indices, bool print_grad) 
{
    if (current_dim >= tensor->ndim) 
    {
        return;
    }

    printf("[");
    for (size_t i = 0; i < tensor->shape[current_dim]; i++) 
    {
        indices[current_dim] = i;

        if (current_dim == tensor->ndim - 1) 
        {
            size_t flat_index = 0;
            for (size_t j = 0; j < tensor->ndim; j++)
            {
                flat_index += indices[j] * tensor->stride[j];
            }

            if (print_grad)
            {
                if (tensor->grad != NULL && tensor->type == TENSOR_TYPE_FLOAT)
                {
                    printf("% .2e", tensor->grad[flat_index]);
                }
                else
                {
                    printf("None");
                }
            }
            else
            {
                if (tensor->type == TENSOR_TYPE_FLOAT)
                {
                    printf("% .2e", tensor->data.f_data[flat_index]);
                }
                else if (tensor->type == TENSOR_TYPE_UINT8)
                {
                    printf("%u", tensor->data.u8_data[flat_index]);
                }
            }
        }
        else 
        {
            print_tensor_array_recursive(tensor, current_dim + 1, indices, print_grad);
        }

        if (i < tensor->shape[current_dim] - 1) 
        {
            printf(", ");
            if (current_dim < tensor->ndim - 1)
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
    printf("type: %s\n", (tensor->type == TENSOR_TYPE_FLOAT) ? "float" : "uint8");
    printf("shape: (");
    for (size_t i = 0; i < tensor->ndim; i++) 
    {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->ndim - 1) 
        {
            printf(", ");
        }
    }
    printf(")\nstride: (");
    for (size_t i = 0; i < tensor->ndim; i++) 
    {
        printf("%zu", tensor->stride[i]);
        if (i < tensor->ndim - 1) 
        {
            printf(", ");
        }
    }
    printf(")\n");

    size_t indices[MAX_DIMS] = {0};
    
    printf("data:\n");
    if ((tensor->type == TENSOR_TYPE_FLOAT && tensor->data.f_data != NULL) ||
        (tensor->type == TENSOR_TYPE_UINT8 && tensor->data.u8_data != NULL))
    {
        print_tensor_array_recursive(tensor, 0, indices, false);
    }
    else
    {
        printf("None");
    }
    printf("\n");

    printf("grad:\n");
    if (tensor->grad != NULL && tensor->type == TENSOR_TYPE_FLOAT) 
    {
        print_tensor_array_recursive(tensor, 0, indices, true);
    }
    else
    {
        printf("None");
    }
    printf("\n");
}