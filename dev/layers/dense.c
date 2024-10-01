#include <time.h>
#include <stdio.h>
#include <cortex.h>

int main() 
{
    pool_init(256 * KB);

    srand((unsigned int)time(NULL));

    // Define input dimensions
    size_t batch_size = 2;
    size_t input_dim = 3;
    size_t output_dim = 4;

    // Create an input tensor with random values
    size_t input_shape[2] = {batch_size, input_dim};
    tensor_t* input_tensor = tensor_rand(input_shape, 2, 1.0f);
    if (input_tensor == NULL) 
    {
        printf("Failed to create input_tensor\n");
        pool_destroy();
        return -1;
    } 
    else 
    {
        print_tensor(input_tensor, "Input tensor");
    }

    // Create a dense layer
    layer_t* dense_layer = dense_create("dense_layer", input_dim, output_dim);
    if (dense_layer == NULL) 
    {
        printf("Failed to create dense_layer\n");
        tensor_destroy(input_tensor);
        pool_destroy();
        return -1;
    }

    // Perform a forward pass through the dense layer
    tensor_t* output_tensor = layer_forward(dense_layer, input_tensor);
    if (output_tensor == NULL) 
    {
        printf("Failed to perform forward pass\n");
        tensor_destroy(input_tensor);
        layer_free(dense_layer);
        pool_destroy();
        return -1;
    } 
    else 
    {
        print_tensor(output_tensor, "Output tensor");
    }

    // Simulate a loss gradient for testing
    for (size_t i = 0; i < output_tensor->size; ++i)
    {
        output_tensor->grad[i] = 1.0f;
    }

    // Perform a backward pass to compute gradients
    if (output_tensor->backward) 
    {
        output_tensor->backward(output_tensor);
    }

    // Retrieve the weights and bias gradients
    dense_parameters_t* params = (dense_parameters_t*)dense_layer->params;
    tensor_t* weights = params->weights;
    tensor_t* bias = params->bias;

    // Print tensors
    print_tensor(weights, "Weights");
    print_tensor(bias, "Bias");
    print_tensor(input_tensor, "Input tensor");

    // Cleanup
    tensor_destroy(input_tensor);
    layer_free(dense_layer);

    printf("Used memory: %zu bytes\n", pool_get_used_memory());
    printf("Free memory: %zu bytes\n", pool_get_free_memory());

    pool_destroy();

    return 0;
}