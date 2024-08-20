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
    // Input, shape: {2, 4}
    float input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int input_shape[] = {2, 4};
    Tensor *input = tensor_from_array(input_data, input_shape, 2);

    // First Dense layer, shape {4, 3}
    Layer *fc1 = dense_create(4, 3);
    float w1[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    float b1[] = {0.5, 0.5, 0.5};
    memcpy(fc1->weights->data, w1, sizeof(w1));
    memcpy(fc1->bias->data, b1, sizeof(b1));

    // Second Dense layer, shape {3, 2}
    Layer *fc2 = dense_create(3, 2);
    float w2[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    float b2[] = {0.1, 0.1};
    memcpy(fc2->weights->data, w2, sizeof(w2));
    memcpy(fc2->bias->data, b2, sizeof(b2));

    // Create the model and add layers
    Layer *layers[] = {fc1, fc2};
    Model *model = model_create(layers, 2);

    // Forward pass
    Tensor *hidden = fc1->forward(fc1, input);
    Tensor *output = fc2->forward(fc2, hidden);

    // Set the gradient of the output tensor
    for (int i = 0; i < output->size; i++)
    {
        output->grad[i] = 1.0;
    }

    // Create an optimizer
    Optimizer *sgd = sgd_create(0.01);

    // Backward pass
    output->backward(output, output->grad);

    // Perform a single optimization step for all parameters in the model
    sgd->step(sgd, model->params, model->num_params);

    // Print results
    print_tensor(output, "output");
    print_tensor(fc2->weights, "weights fc2 (after update)");
    print_tensor(fc2->bias, "bias fc2 (after update)");
    print_tensor(fc1->weights, "weights fc1 (after update)");
    print_tensor(fc1->bias, "bias fc1 (after update)");

    // Free memory
    tensor_free(input);
    optimizer_free(sgd);
    model_free(model);

    return 0;
}