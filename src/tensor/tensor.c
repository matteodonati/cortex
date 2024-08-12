#include <stdlib.h>
#include "tensor/tensor.h"

Tensor* create(int *shape, int ndim) 
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int *)malloc(ndim * sizeof(int));
    for(int i = 0; i < ndim; i++) 
    {
        tensor->shape[i] = shape[i];
    }
    tensor->size = 1;
    tensor->stride = (int *)malloc(ndim * sizeof(int));
    tensor->stride[tensor->ndim - 1] = 1;
    for(int i = tensor->ndim - 1; i >= 0; --i) 
    {
        tensor->size *= tensor->shape[i];
        if (i > 0) 
        {
            tensor->stride[i - 1] = tensor->stride[i] * tensor->shape[i];
        }
    }
    tensor->data = (float *)malloc(tensor->size * sizeof(float));
    return tensor;
}

void free_tensor(Tensor *tensor) 
{
    if(tensor) 
    {
        free(tensor->data);
        free(tensor->shape);
        free(tensor->stride);
        free(tensor);
    }
}
