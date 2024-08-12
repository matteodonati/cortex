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

Tensor* tensor_from_array(float *array, int *shape, int ndim);
Tensor* tensor_rand(int *shape, int ndim);
Tensor* tensor_zeros(int *shape, int ndim);
Tensor* tensor_ones(int *shape, int ndim);
Tensor* tensor_eye(int size);
void free_tensor(Tensor *tensor);

#endif