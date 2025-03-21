#include <time.h>
#include <stdio.h>
#include <cortex.h>

int main() 
{
    pool_init(256 * KB);

    srand((unsigned int)time(NULL));

    size_t shape1[2] = {2, 3};
    size_t shape2[3] = {2, 2, 2};
    size_t ndim1 = 2;
    size_t ndim2 = 3;

    // tensor_rand
    tensor_t* tensor_rand_example = tensor_rand(shape1, ndim1, 1.0f);
    if (tensor_rand_example == NULL) 
    {
        printf("Failed to create tensor_rand_example\n");
    } else 
    {
        print_tensor(tensor_rand_example, "tensor_rand_example");
    }

    // tensor_zeros
    tensor_t* tensor_zeros_example = tensor_zeros(shape2, ndim2);
    if (tensor_zeros_example == NULL) 
    {
        printf("Failed to create tensor_zeros_example\n");
    } else 
    {
        print_tensor(tensor_zeros_example, "tensor_zeros_example");
    }

    // tensor_ones
    tensor_t* tensor_ones_example = tensor_ones(shape1, ndim1);
    if (tensor_ones_example == NULL) 
    {
        printf("Failed to create tensor_ones_example\n");
    } else 
    {
        print_tensor(tensor_ones_example, "tensor_ones_example");
    }

    // tensor_full
    float fill_value = 3.14f;
    tensor_t* tensor_full_example = tensor_full(shape2, ndim2, fill_value);
    if (tensor_full_example == NULL) 
    {
        printf("Failed to create tensor_full_example\n");
    } else 
    {
        print_tensor(tensor_full_example, "tensor_full_example");
    }

    // tensor_from_array
    float array_data[6] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f};
    tensor_t* tensor_from_array_example = tensor_from_array(array_data, shape1, ndim1);
    if (tensor_from_array_example == NULL) 
    {
        printf("Failed to create tensor_from_array_example\n");
    } else 
    {
        print_tensor(tensor_from_array_example, "tensor_from_array_example");
    }

    // tensor_like
    tensor_t* tensor_like_example = tensor_like(tensor_rand_example);
    if (tensor_like_example == NULL) 
    {
        printf("Failed to create tensor_like_example\n");
    } else 
    {
        print_tensor(tensor_like_example, "tensor_like_example");
    }

    // tensor_clone
    tensor_t* tensor_clone_example = tensor_clone(tensor_full_example);
    if (tensor_clone_example == NULL) 
    {
        printf("Failed to create tensor_clone_example\n");
    } else 
    {
        print_tensor(tensor_clone_example, "tensor_clone_example");
    }

    // tensor_add
    tensor_t* tensor_add_example = tensor_add(tensor_rand_example, tensor_ones_example);
    if (tensor_add_example == NULL) 
    {
        printf("Failed to create tensor_add_example\n");
    } else 
    {
        print_tensor(tensor_add_example, "tensor_add_example");
    }

    // tensor_reshape
    size_t new_shape[3] = {1, 3, 2};
    size_t new_ndim = 3;
    tensor_t* tensor_reshape_example = tensor_reshape(tensor_rand_example, new_shape, new_ndim);
    if (tensor_reshape_example == NULL) 
    {
        printf("Failed to create tensor_reshape_example\n");
    } else 
    {
        print_tensor(tensor_reshape_example, "tensor_reshape_example");
    }

    printf("Used memory before destroy: %zu bytes\n", pool_get_used_memory());
    printf("Free memory before destroy: %zu bytes\n", pool_get_free_memory());

    tensor_destroy(tensor_rand_example);
    tensor_destroy(tensor_zeros_example);
    tensor_destroy(tensor_ones_example);
    tensor_destroy(tensor_full_example);
    tensor_destroy(tensor_from_array_example);
    tensor_destroy(tensor_like_example);
    tensor_destroy(tensor_clone_example);
    tensor_destroy(tensor_add_example);
    tensor_destroy(tensor_reshape_example);

    printf("Used memory after destroy: %zu bytes\n", pool_get_used_memory());
    printf("Free memory after destroy: %zu bytes\n", pool_get_free_memory());

    pool_destroy();

    return 0;
}