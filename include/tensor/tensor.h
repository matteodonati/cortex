#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>
#include "utils/status/status.h"

#define MAX_DIMS 4

typedef struct tensor
{
    size_t ndim;
    size_t size;
    size_t shape[MAX_DIMS];
    size_t stride[MAX_DIMS];
    bool frozen;
    float* data;
    float* grad;
    struct tensor* grad_a;
    struct tensor* grad_b;
    void (*backward)(struct tensor* self);
} tensor_t;

tensor_status_code_t tensor_destroy(tensor_t* tensor);
tensor_t* tensor_from_array(const float *array, const size_t *shape, size_t ndim);
tensor_t* tensor_rand(const size_t *shape, size_t ndim, float limit);
tensor_t* tensor_zeros(const size_t *shape, size_t ndim);
tensor_t* tensor_ones(const size_t *shape, size_t ndim);
tensor_t* tensor_full(const size_t *shape, size_t ndim, float value);
tensor_t* tensor_like(const tensor_t *a);
tensor_t* tensor_clone(const tensor_t *a);

#endif