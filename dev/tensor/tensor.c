#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cortex.h>

void print_tensors()
{
    int ndim1 = 2;
    int ndim2 = 3;

    int shape1[2] = {2, 3};
    int shape2[3] = {10, 10, 10};

    // Create a tensor from an array
    float array[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor *tensor1 = tensor_from_array(array, shape1, ndim1);
    printf("Tensor 1 (from array):\n");
    print_tensor_shape(tensor1);
    print_tensor_data(tensor1);
    printf("\n");

    // Create a random tensor
    Tensor *tensor2 = tensor_rand(shape1, ndim1);
    printf("Tensor 2 (random):\n");
    print_tensor_shape(tensor2);
    print_tensor_data(tensor2);
    printf("\n");

    // Create a tensor filled with zeros
    Tensor *tensor3 = tensor_zeros(shape1, ndim1);
    printf("Tensor 3 (zeros):\n");
    print_tensor_shape(tensor3);
    print_tensor_data(tensor3);
    printf("\n");

    // Create a tensor filled with ones
    Tensor *tensor4 = tensor_ones(shape1, ndim1);
    printf("Tensor 4 (ones):\n");
    print_tensor_shape(tensor4);
    print_tensor_data(tensor4);
    printf("\n");

    // Create a tensor filled with a custom value
    Tensor *tensor5 = tensor_full(shape1, ndim1, 2.0);
    printf("Tensor 5 (full):\n");
    print_tensor_shape(tensor5);
    print_tensor_data(tensor5);
    printf("\n");

    // Create a tensor filled with ones
    Tensor *tensor6 = tensor_rand(shape2, ndim2);
    printf("Tensor 6 (n-dimensional):\n");
    print_tensor_shape(tensor6);
    print_tensor_data(tensor6);

    // Free tensors
    tensor_free(tensor1);
    tensor_free(tensor2);
    tensor_free(tensor3);
    tensor_free(tensor4);
    tensor_free(tensor5);
    tensor_free(tensor6);
}

int main()
{
    srand(time(NULL));
    print_tensors();
    return 0;
}
