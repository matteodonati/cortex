#include <stdio.h>
#include <assert.h>
#include <cortex.h>

int main() 
{
    // Initialize memory pool with initial size (e.g., 1MB). This means that for each MB malloc is called twice.
    pool_init(1 * MB);

    // Define tensor shapes
    size_t shape[2] = {16, 16};
    
    // Print initial memory status
    printf("Initial used memory: %zu bytes\n", pool_get_used_memory());
    printf("Initial free memory: %zu bytes\n", pool_get_free_memory());

    // Create first tensor
    tensor_t* tensor1 = tensor_create(2, shape);
    assert(tensor1 != NULL);

    // Print memory status after first allocation
    printf("Used memory after first tensor creation: %zu bytes\n", pool_get_used_memory());
    printf("Free memory after first tensor creation: %zu bytes\n", pool_get_free_memory());

    // Create second tensor
    tensor_t* tensor2 = tensor_create(2, shape);
    assert(tensor2 != NULL);

    // Print memory status after second allocation
    printf("Used memory after second tensor creation: %zu bytes\n", pool_get_used_memory());
    printf("Free memory after second tensor creation: %zu bytes\n", pool_get_free_memory());

    // Free the first tensor
    tensor_destroy(tensor1);

    // Print memory status after freeing first tensor
    printf("Used memory after freeing first tensor: %zu bytes\n", pool_get_used_memory());
    printf("Free memory after freeing first tensor: %zu bytes\n", pool_get_free_memory());

    // Free the second tensor
    tensor_destroy(tensor2);

    // Print memory status after freeing second tensor
    printf("Used memory after freeing second tensor: %zu bytes\n", pool_get_used_memory());
    printf("Free memory after freeing second tensor: %zu bytes\n", pool_get_free_memory());

    // Create third tensor
    tensor_t* tensor3 = tensor_create(2, shape);
    assert(tensor3 != NULL);

    // Print memory status after third allocation
    printf("Used memory after third tensor creation: %zu bytes\n", pool_get_used_memory());
    printf("Free memory after third tensor creation: %zu bytes\n", pool_get_free_memory());

    // Free the third tensor
    tensor_destroy(tensor3);

    // Print memory status after freeing third tensor
    printf("Used memory after freeing third tensor: %zu bytes\n", pool_get_used_memory());
    printf("Free memory after freeing third tensor: %zu bytes\n", pool_get_free_memory());

    // Destroy the memory pool
    pool_destroy();

    return 0;
}