#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cortex.h>

int main() 
{
    // Input data (batch_size, in_channels, height, width)
    float x_data[] = {
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    int x_shape[] = {1, 1, 4, 4}; // b, c, h, w
    Tensor *x = tensor_from_array("x", x_data, x_shape, 4);

    // Conv2D layer with 1 input channel, 1 output channel, 3x3 kernel size
    int kernel_size[] = {3, 3};
    int stride[] = {1, 1};
    int padding[] = {0, 0};
    int groups = 1;
    Layer *conv1 = conv2d_create("conv1", 1, 1, kernel_size, stride, padding, groups);
    Conv2DParameters *conv_params = (Conv2DParameters *)(conv1->params);

    // Create the model and add layers
    int num_layers = 1;
    Layer *layers[] = {conv1};
    Model *model = model_create(layers, num_layers);

    // Set weights and bias
    float w[] = {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9
    };
    float b[] = {1.0};
    memcpy(conv_params->weights->data, w, sizeof(w));
    memcpy(conv_params->bias->data, b, sizeof(b));

    // Forward pass
    Tensor *y_pred = layer_forward(conv1, x);

    // Print output tensors
    print_tensor(y_pred, "y_pred");

    // Free memory
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