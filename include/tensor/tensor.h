#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

struct Tensor;

typedef struct TensorOpsUtils
{
    struct Tensor **cached_tensors;
    float *cached_floats;
    int *cached_ints;
    struct Tensor *cached_tensor;
    float cached_float;
    int cached_int;
} TensorOpsUtils;

typedef struct Tensor 
{
    char *name;
    float *data;
    float *grad;
    int *shape;
    int *stride;
    struct Tensor *grad_a;
    struct Tensor *grad_b;
    void (*backward)(struct Tensor *self);
    TensorOpsUtils ops_utils;
    int ndim;
    int size;
    bool frozen;
} Tensor;

Tensor* initialize_tensor(const char *name, int *shape, int ndim);
Tensor* tensor_from_array(const char *name, float *array, int *shape, int ndim);
Tensor* tensor_rand(const char *name, int *shape, int ndim, float limit);
Tensor* tensor_zeros(const char *name, int *shape, int ndim);
Tensor* tensor_ones(const char *name, int *shape, int ndim);
Tensor* tensor_full(const char *name, int *shape, int ndim, float value);
Tensor* tensor_like(const char *name, Tensor *a);
Tensor* tensor_clone(const char *name, Tensor *a);
void tensor_free(Tensor *tensor);

#endif