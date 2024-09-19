#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include "utils/status/status.h"

#define KB 1024
#define MB 1024 * 1024
#define ALIGNMENT 16
#define ALIGN_UP(x) (((x) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

typedef struct MemoryBlock 
{
    size_t size;
    struct MemoryBlock* next;
} MemoryBlock;

typedef struct MemoryPool 
{
    uint8_t* pool;
    size_t size;
    size_t used;
    MemoryBlock* free_list;
    struct MemoryPool* next;
} MemoryPool;

extern MemoryPool* global_memory_pool;

memory_pool_status_code_t pool_init(size_t initial_size);
memory_pool_status_code_t pool_destroy();
void* pool_alloc(size_t size);
memory_pool_status_code_t pool_free(void* ptr);
size_t pool_get_used_memory();
size_t pool_get_free_memory();

#endif