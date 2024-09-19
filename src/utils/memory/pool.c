#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "utils/memory/pool.h"

MemoryPool* global_memory_pool = NULL;

static MemoryPool* pool_create(size_t size) 
{
    MemoryPool* pool = (MemoryPool*)malloc(sizeof(MemoryPool));
    if (!pool)
    {
        return NULL;
    }

    pool->pool = (uint8_t*)malloc(size);
    if (!pool->pool) 
    {
        free(pool);
        return NULL;
    }

    pool->size = size;
    pool->used = 0;
    pool->free_list = NULL;
    pool->next = NULL;

    return pool;
}

static memory_pool_status_code_t pool_expand(size_t required_size) 
{
    size_t new_pool_size = (global_memory_pool->size > required_size) ? global_memory_pool->size : required_size;

    MemoryPool* new_pool = pool_create(new_pool_size);
    if (new_pool == NULL) 
    {
        return POOL_EXPAND_FAILURE;
    }

    MemoryPool* pool = global_memory_pool;
    while (pool->next) 
    {
        pool = pool->next;
    }
    pool->next = new_pool;

    return POOL_EXPAND_SUCCESS;
}

memory_pool_status_code_t pool_init(size_t initial_size) 
{
    global_memory_pool = pool_create(initial_size);
    if (global_memory_pool == NULL)
    {
        return POOL_CREATION_FAILURE;
    }
    return POOL_CREATION_SUCCESS;
}

memory_pool_status_code_t pool_destroy()
{
    MemoryPool* pool = global_memory_pool;
    if (pool == NULL)
    {
        return POOL_DESTROY_FAILURE;
    }

    while (pool) 
    {
        MemoryPool* next_pool = pool->next;
        free(pool->pool);
        free(pool);
        pool = next_pool;
    }
    global_memory_pool = NULL;

    return POOL_DESTROY_SUCCESS;
}

void* pool_alloc(size_t size) 
{
    size = ALIGN_UP(size);
    size_t total_size = size + sizeof(MemoryBlock);

    // Check the free list across all pools for a suitable block
    MemoryPool* pool = global_memory_pool;
    while (pool) 
    {
        MemoryBlock** prev = &pool->free_list;
        MemoryBlock* current = pool->free_list;
        while (current) 
        {
            if (current->size >= size) 
            {
                *prev = current->next;
                pool->used += total_size;
                return (void*)(current + 1);
            }
            prev = &current->next;
            current = current->next;
        }
        pool = pool->next;
    }

    // Allocate memory from the current pool
    pool = global_memory_pool;
    while (pool) 
    {
        if (pool->used + total_size <= pool->size) 
        {
            MemoryBlock* block = (MemoryBlock*)(pool->pool + pool->used);
            block->size = size;
            pool->used += total_size;
            return (void*)(block + 1);
        }
        pool = pool->next;
    }

    // Create a new pool
    if (pool_expand(total_size) == POOL_EXPAND_FAILURE) 
    {
        return NULL;
    }
    pool = global_memory_pool;
    while (pool->next) 
    {
        pool = pool->next;
    }
    MemoryBlock* block = (MemoryBlock*)(pool->pool + pool->used);
    block->size = size;
    pool->used += total_size;

    return (void*)(block + 1);
}

memory_pool_status_code_t pool_free(void* ptr) 
{
    if (ptr == NULL)
    {
        return POOL_FREE_FAILURE;
    }

    // Retrieve the block header
    MemoryBlock* block = ((MemoryBlock*)ptr) - 1;
    uintptr_t block_addr = (uintptr_t)block;

    // Add the block to the free list in the corresponding pool
    MemoryPool* pool = global_memory_pool;
    while (pool) 
    {
        uintptr_t pool_start = (uintptr_t)pool->pool;
        uintptr_t pool_end = pool_start + pool->size;

        if (block_addr >= pool_start && block_addr < pool_end) 
        {
            block->next = pool->free_list;
            pool->free_list = block;
            pool->used -= (block->size + sizeof(MemoryBlock));
            return POOL_FREE_SUCCESS;
        }

        pool = pool->next;
    }

    return POOL_FREE_FAILURE;
}

size_t pool_get_used_memory(void) 
{
    size_t total_used = 0;
    MemoryPool* pool = global_memory_pool;
    while (pool) 
    {
        total_used += pool->used;
        pool = pool->next;
    }
    return total_used;
}

size_t pool_get_free_memory(void) 
{
    size_t total_free = 0;
    MemoryPool* pool = global_memory_pool;
    while (pool) 
    {
        total_free += pool->size - pool->used;
        pool = pool->next;
    }
    return total_free;
}