#ifndef STATUS_H
#define STATUS_H

typedef enum memory_pool_status_code
{
    POOL_CREATION_SUCCESS,
    POOL_CREATION_FAILURE,
    POOL_EXPAND_SUCCESS,
    POOL_EXPAND_FAILURE,
    POOL_DESTROY_SUCCESS,
    POOL_DESTROY_FAILURE,
    POOL_FREE_SUCCESS,
    POOL_FREE_FAILURE
} memory_pool_status_code_t;

typedef enum tensor_status_code
{
    TENSOR_DESTROY_SUCCESS,
    TENSOR_DESTROY_FAILURE
} tensor_status_code_t;

#endif