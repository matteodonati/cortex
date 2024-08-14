#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor {
    float *data;
    float *grad;
    int *shape;
    int *stride;
    int *axes;
    int ndim;
    int size;
    void (*backward)(struct Tensor *self, float *grad);
    struct Tensor *grad_a;
    struct Tensor *grad_b;
} Tensor;


Tensor* tensor_from_array(float *array, int *shape, int ndim);
Tensor* tensor_rand(int *shape, int ndim);
Tensor* tensor_zeros(int *shape, int ndim);
Tensor* tensor_ones(int *shape, int ndim);
Tensor* tensor_full(int *shape, int ndim, float value);
Tensor* allocate_tensor_with_same_shape(Tensor *a);
void tensor_free(Tensor *tensor);

#endif