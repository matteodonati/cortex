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

    // Conv2D layer with 3 input channels, 2 output channels, 3x3 kernel size
    int kernel_size[] = {3, 3};
    int stride[] = {1, 1};
    int padding[] = {0, 0};
    int out_channels = 2;
    Layer *conv1 = conv2d_create("conv1", 3, out_channels, kernel_size, stride, padding);
    Conv2DParameters *conv_params = (Conv2DParameters *)(conv1->params);

    // BatchNorm2D layer with 2 output channels (same as Conv2D output)
    Layer *bn1 = batchnorm2d_create("bn1", out_channels, 1e-5, 0.1);

    // Create the model and add layers
    int num_layers = 2;
    Layer *layers[] = {conv1, bn1};
    Model *model = model_create(layers, num_layers);

    // Set weights and biases for the two Conv2D kernels
    float w[] = {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        
        0.1, -0.2, 0.3,
        -0.4, 0.5, -0.6,
        0.7, -0.8, 0.9,

        0.2, 0.1, 0.3,
        0.6, 0.4, 0.5,
        0.8, 0.7, 0.9,

        -0.1, -0.2, -0.3,
        -0.4, -0.5, -0.6,
        -0.7, -0.8, -0.9,

        -0.1, 0.2, -0.3,
        0.4, -0.5, 0.6,
        -0.7, 0.8, -0.9,

        -0.2, -0.1, -0.3,
        -0.6, -0.4, -0.5,
        -0.8, -0.7, -0.9
    };
    float b[] = {1.0, 0.5};
    memcpy(conv_params->weights->data, w, sizeof(w));
    memcpy(conv_params->bias->data, b, sizeof(b));

    // Set training flag to true for all layers
    model_train(model);

    // Forward pass through Conv2D and BatchNorm2D
    Tensor *y_conv = layer_forward(conv1, x);
    Tensor *y_pred = layer_forward(bn1, y_conv);

    // Backward pass (assuming some loss function with gradient 1.0 for simplicity)
    Tensor *loss_grad = tensor_ones(NULL, y_pred->shape, y_pred->ndim);
    memcpy(y_pred->grad, loss_grad->data, loss_grad->size * sizeof(float));

    backward(y_pred);

    // Print output tensors
    print_tensor(y_conv, "y_conv");
    print_tensor(y_pred, "y_pred");

    // Print gradients for Conv2D weights, bias, and BatchNorm gamma and beta
    print_tensor(conv_params->weights, "conv1.weight");
    print_tensor(conv_params->bias, "conv1.bias");
    BatchNorm2DParameters *bn_params = (BatchNorm2DParameters *)(bn1->params);
    print_tensor(bn_params->gamma, "bn1.gamma");
    print_tensor(bn_params->beta, "bn1.beta");

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