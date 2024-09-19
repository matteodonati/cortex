#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <assert.h>
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

// tensor_create should be static.
// TODO: implement other initialization functions.
tensor_t* tensor_create(size_t dim, const size_t shape[]);
tensor_status_code_t tensor_destroy(tensor_t* tensor);

#endif