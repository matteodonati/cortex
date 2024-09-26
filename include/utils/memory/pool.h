#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include "utils/status/status.h"

#define KB 1024
#define MB 1024 * 1024

typedef struct memory_block 
{
    size_t size;
    struct memory_block* next;
} memory_block_t;

typedef struct memory_pool 
{
    uint8_t* pool;
    size_t size;
    size_t used;
    memory_block_t* free_list;
    struct memory_pool* next;
} memory_pool_t;

extern memory_pool_t* global_memory_pool;

memory_pool_status_code_t pool_init(size_t initial_size);
memory_pool_status_code_t pool_destroy();
void* pool_alloc(size_t size);
memory_pool_status_code_t pool_free(void* ptr);
size_t pool_get_used_memory();
size_t pool_get_free_memory();

#endif