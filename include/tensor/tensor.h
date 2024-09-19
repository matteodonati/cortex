#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include "utils/status/status.h"

#define MAX_DIMS 4

typedef enum tensor_type 
{
    TENSOR_TYPE_FLOAT,
    TENSOR_TYPE_UINT8
} tensor_type_t;

typedef struct tensor
{
    tensor_type_t type;
    size_t ndim;
    size_t size;
    size_t shape[MAX_DIMS];
    size_t stride[MAX_DIMS];
    bool frozen;
    union {
        float* f_data;
        uint8_t* u8_data;
    } data;
    float* grad;
    struct tensor* grad_a;
    struct tensor* grad_b;
    void (*backward)(struct tensor* self);
} tensor_t;

// tensor_create should be static.
// TODO: implement other initialization functions.
tensor_t* tensor_create(size_t dim, const size_t shape[], const tensor_type_t type);
tensor_status_code_t tensor_destroy(tensor_t* tensor);

#endif