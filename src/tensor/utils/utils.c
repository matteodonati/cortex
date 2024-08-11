#include <stdio.h>
#include "tensor/tensor.h"
#include "tensor/utils/utils.h"

void print_shape(Tensor *tensor) 
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