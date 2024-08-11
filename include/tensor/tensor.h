#ifndef TENSOR_H
#define TENSOR_H

typedef struct 
{
    float *data;
    int *shape;
    int *stride;
    int ndim;
    int size;
} Tensor;

Tensor* create(int *shape, int ndim);
void free_tensor(Tensor *tensor);

#endif