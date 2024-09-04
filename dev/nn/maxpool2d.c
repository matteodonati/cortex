#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cortex.h>

int main() 
{
    // Input data (batch_size, in_channels, height, width)
    float x_data[] = {
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        
        1.5,  2.5,  3.5,  4.5,
        5.5,  6.5,  7.5,  8.5,
        9.5,  10.5, 11.5, 12.5,
        13.5, 14.5, 15.5, 16.5,
        
        2.0,  3.0,  4.0,  5.0,
        6.0,  7.0,  8.0,  9.0,
        10.0, 11.0, 12.0, 13.0,
        14.0, 15.0, 16.0, 17.0
    };
    int x_shape[] = {1, 3, 4, 4}; // b, c, h, w
    Tensor *x = tensor_from_array("x", x_data, x_shape, 4);

    // MaxPool2D layer with 3 input channels, kernel size 3x3, stride 1x1, and padding 0
    int kernel_size[] = {2, 2};
    int stride[] = {2, 2};
    int padding[] = {0, 0};
    Layer *pool1 = maxpool2d_create("pool1", kernel_size, stride, padding);

    // Create the model and add layers
    int num_layers = 1;
    Layer *layers[] = {pool1};
    Model *model = model_create(layers, num_layers);

    // Forward pass
    Tensor *y_pred = layer_forward(pool1, x);

    // Print output tensor
    print_tensor(y_pred, "y_pred");

    // Backward pass (assuming some loss function with gradient 1.0 for simplicity)
    Tensor *loss_grad = tensor_ones(NULL, y_pred->shape, y_pred->ndim);
    memcpy(y_pred->grad, loss_grad->data, loss_grad->size * sizeof(float));

    backward(y_pred);

    // Print gradients for input tensor
    print_tensor(x, "input.grad");

    // Free memory
    tensor_free(loss_grad);
    for (int i = 0; i < num_layers; i++) 
    {
        for (int j = 0; j < layers[i]->tensor_count; j++) 
        {
            tensor_free(layers[i]->tensors[j]);
        }
        free(layers[i]->tensors);
    }
    tensor_free(x);
    model_free(model);

    return 0;
}