#include <time.h>
#include <stdio.h>
#include <string.h>
#include <cortex.h>

int main() 
{
    pool_init(256 * KB);

    // Define input dimensions
    size_t batch_size = 2;
    size_t input_dim = 3;
    size_t output_dim = 4;

    // Create an input tensor with specific values
    size_t input_shape[2] = {batch_size, input_dim};
    tensor_t* input_tensor = tensor_zeros(input_shape, 2);
    if (input_tensor == NULL) 
    {
        printf("Failed to create input_tensor\n");
        pool_destroy();
        return -1;
    }
    input_tensor->data[0] = 1.0f;
    input_tensor->data[1] = 2.0f;
    input_tensor->data[2] = 3.0f;
    input_tensor->data[3] = 4.0f;
    input_tensor->data[4] = 5.0f;
    input_tensor->data[5] = 6.0f;
    print_tensor(input_tensor, "input");

    // Create a dense layer
    layer_t* dense_layer = dense_create("dense_layer", input_dim, output_dim);
    if (dense_layer == NULL) 
    {
        printf("Failed to create dense_layer\n");
        tensor_destroy(input_tensor);
        pool_destroy();
        return -1;
    }

    // Set specific values for weights and biases
    dense_parameters_t* params = (dense_parameters_t*)dense_layer->params;

    // Weights tensor
    float weights_values[] = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f
    };
    memcpy(params->weights->data, weights_values, sizeof(weights_values));

    // Bias tensor
    float bias_values[] = {0.1f, 0.2f, 0.3f, 0.4f};
    memcpy(params->bias->data, bias_values, sizeof(bias_values));

    // Perform a forward pass through the dense layer
    tensor_t* output_tensor = layer_forward(dense_layer, input_tensor);
    if (output_tensor == NULL) 
    {
        printf("Failed to perform forward pass\n");
        tensor_destroy(input_tensor);
        layer_destroy(dense_layer);
        pool_destroy();
        return -1;
    } 
    else 
    {
        print_tensor(output_tensor, "output");
    }

    // Perform a backward pass to compute gradients
    for (size_t i = 0; i < output_tensor->size; ++i)
    {
        output_tensor->grad[i] = 1.0f;
    }
    if (output_tensor->backward) 
    {
        output_tensor->backward(output_tensor);
    }

    // Print weights, biases
    print_tensor(params->weights, "weights");
    print_tensor(params->bias, "biases");

    // Cleanup
    tensor_destroy(input_tensor);
    layer_destroy(dense_layer);

    printf("Used memory: %zu bytes\n", pool_get_used_memory());
    printf("Free memory: %zu bytes\n", pool_get_free_memory());

    pool_destroy();

    return 0;
}