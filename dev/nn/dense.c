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

#include <stdio.h>
#include <stdlib.h>
#include "tensor/tensor.h"
#include "nn/dense.h"
#include "optim/optim.h"

// Main function to stack two Dense layers and perform forward and backward passes
int main() 
{
    // Input Tensor (shape: 2x4)
    float input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int input_shape[] = {2, 4};
    Tensor *input = tensor_from_array(input_data, input_shape, 2);

    // First Dense Layer (4x3)
    Layer *dense_layer1 = dense_create(4, 3);
    float weight_data1[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    float bias_data1[] = {0.5, 0.5, 0.5};
    memcpy(dense_layer1->weights->data, weight_data1, sizeof(weight_data1));
    memcpy(dense_layer1->bias->data, bias_data1, sizeof(bias_data1));

    // Second Dense Layer (3x2)
    Layer *dense_layer2 = dense_create(3, 2);
    float weight_data2[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    float bias_data2[] = {0.1, 0.1};
    memcpy(dense_layer2->weights->data, weight_data2, sizeof(weight_data2));
    memcpy(dense_layer2->bias->data, bias_data2, sizeof(bias_data2));

    // Forward Pass
    Tensor *hidden_output = dense_layer1->forward(dense_layer1, input);
    Tensor *output = dense_layer2->forward(dense_layer2, hidden_output);

    // Set the gradient of the output tensor
    for (int i = 0; i < output->size; i++)
    {
        output->grad[i] = 1.0;
    }

    // Backward pass starting from the output tensor
    output->backward(output, output->grad);

    // Create an optimizer and update parameters
    Optimizer *sgd = create_sgd_optimizer(0.01);
    sgd->update(sgd, dense_layer2->weights, dense_layer2->bias);
    sgd->update(sgd, dense_layer1->weights, dense_layer1->bias);

    // Print results
    print_tensor(output, "Output");
    print_tensor(dense_layer2->weights, "Weights Layer 2 (after update)");
    print_tensor(dense_layer2->bias, "Bias Layer 2 (after update)");
    print_tensor(dense_layer1->weights, "Weights Layer 1 (after update)");
    print_tensor(dense_layer1->bias, "Bias Layer 1 (after update)");

    // Free memory
    tensor_free(input);
    optimizer_free(sgd);
    layer_free(dense_layer1);
    layer_free(dense_layer2);

    return 0;
}