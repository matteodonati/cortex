#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

struct Tensor;

typedef struct TensorOpsUtils
{
    int cached_int;
    int *cached_ints;
    float cached_float;
    float *cached_floats;
    struct Tensor *cached_tensor;
    struct Tensor **cached_tensors;
} TensorOpsUtils;

typedef struct Tensor 
{
    char *name;
    float *data;
    float *grad;
    int *shape;
    int *stride;
    int ndim;
    int size;
    bool frozen;
    TensorOpsUtils ops_utils;
    void (*backward)(struct Tensor *self);
    struct Tensor *grad_a;
    struct Tensor *grad_b;
} Tensor;

Tensor* initialize_tensor(const char *name, int *shape, int ndim);
Tensor* tensor_from_array(const char *name, float *array, int *shape, int ndim);
Tensor* tensor_rand(const char *name, int *shape, int ndim, float limit);
Tensor* tensor_zeros(const char *name, int *shape, int ndim);
Tensor* tensor_ones(const char *name, int *shape, int ndim);
Tensor* tensor_full(const char *name, int *shape, int ndim, float value);
Tensor* tensor_like(const char *name, Tensor *a);
void tensor_free(Tensor *tensor);

#endif