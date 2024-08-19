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
    // Input
    float input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int input_shape[] = {2, 4};  // Batch size = 2, input features = 4
    Tensor *input = tensor_from_array(input_data, input_shape, 2);

    // Dense layer
    Layer *dense_layer = dense_create(4, 3);
    float weight_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    float bias_data[] = {0.5, 0.5, 0.5};
    memcpy(dense_layer->weights->data, weight_data, sizeof(weight_data));
    memcpy(dense_layer->bias->data, bias_data, sizeof(bias_data));

    // Forward pass
    Tensor *output = dense_layer->forward(dense_layer, input);

    // Define a dummy gradient for the backward pass
    for (int i = 0; i < output->size; i++)
    {
        output->grad[i] = 1.0;
    }

    print_tensor(output, "Output");
    print_tensor(dense_layer->weights, "Weights (before update)");
    print_tensor(dense_layer->bias, "Bias (before update)");

    // Backward pass
    dense_layer->backward(dense_layer, output->grad);
    Optimizer *sgd = create_sgd_optimizer(0.01);
    sgd->update(sgd, dense_layer->weights, dense_layer->bias);
    print_tensor(dense_layer->weights, "Weights (after update)");
    print_tensor(dense_layer->bias, "Bias (after update)");

    // Free memory
    tensor_free(input);
    tensor_free(output);
    optimizer_free(sgd);
    layer_free(dense_layer);

    return 0;
}