#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cortex.h>

void print_tensor(Tensor *tensor, const char *name) {
    printf("Tensor %s:\n", name);
    printf("shape:\n");
    print_tensor_shape(tensor);
    printf("data:\n");
    print_tensor_data(tensor);
    printf("grad:\n");
    print_tensor_grad(tensor);
    printf("\n");
}

int main() 
{
    // Example input tensor
    int shape[] = {2, 3};
    float data[] = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
    Tensor *input = tensor_from_array("input", data, shape, 2);

    // Apply relu activation
    Tensor *relu_output = relu_f(input);
    print_tensor(relu_output, "ReLU output");

    // Apply sigmoid activation
    Tensor *sigmoid_output = sigmoid_f(input);
    print_tensor(sigmoid_output, "Sigmoid output");

    // Apply tanh activation
    Tensor *tanh_output = tanh_f(input);
    print_tensor(tanh_output, "Tanh output");

    // Apply softmax activation
    Tensor *softmax_output = softmax_f(input, 0);
    print_tensor(softmax_output, "Softmax output");

    // Backward pass
    for (int i = 0; i < relu_output->size; i++) 
    {
        relu_output->grad[i] = 1.0;
    }
    for (int i = 0; i < sigmoid_output->size; i++) 
    {
        sigmoid_output->grad[i] = 1.0;
    }
    for (int i = 0; i < tanh_output->size; i++) 
    {
        tanh_output->grad[i] = 1.0;
    }
    for (int i = 0; i < softmax_output->size; i++) 
    {
        softmax_output->grad[i] = 1.0;
    }

    // Perform backward pass for relu
    tensor_backward(relu_output);
    print_tensor(input, "ReLU backward");

    // Perform backward pass for sigmoid
    tensor_backward(sigmoid_output);
    print_tensor(input, "Sigmoid backward");

    // Perform backward pass for tanh
    tensor_backward(tanh_output);
    print_tensor(input, "Tanh backward");

    // Perform backward pass for softmax
    tensor_backward(softmax_output);
    print_tensor(input, "Softmax backward");

    // Free memory
    tensor_free(input);
    tensor_free(relu_output);
    tensor_free(sigmoid_output);
    tensor_free(tanh_output);
    tensor_free(softmax_output);

    return 0;
}