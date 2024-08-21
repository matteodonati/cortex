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
    Tensor *input = tensor_from_array("input", input_data, input_shape, 2);

    // First Dense layer, shape {4, 3}
    Layer *fc1 = dense_create("fc1", 4, 3);
    float w1[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    float b1[] = {0.5, 0.5, 0.5};
    memcpy(fc1->weights->data, w1, sizeof(w1));
    memcpy(fc1->bias->data, b1, sizeof(b1));

    // Second Dense layer, shape {3, 2}
    Layer *fc2 = dense_create("fc2", 3, 2);
    float w2[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    float b2[] = {0.1, 0.1};
    memcpy(fc2->weights->data, w2, sizeof(w2));
    memcpy(fc2->bias->data, b2, sizeof(b2));

    // Third Dense layer, shape {2, 2}
    Layer *fc3 = dense_create("fc3", 2, 2);
    float w3[] = {0.1, 0.2, 0.3, 0.4};
    float b3[] = {0.1, 0.1};
    memcpy(fc3->weights->data, w3, sizeof(w3));
    memcpy(fc3->bias->data, b3, sizeof(b3));

    // Fourth Dense layer, shape {2, 1}
    Layer *fc4 = dense_create("fc4", 2, 1);
    float w4[] = {0.1, 0.2};
    float b4[] = {0.1};
    memcpy(fc4->weights->data, w4, sizeof(w4));
    memcpy(fc4->bias->data, b4, sizeof(b4));

    // Create the model and add layers
    Layer *layers[] = {fc1, fc2, fc3, fc4};
    Model *model = model_create(layers, 4);

    // Forward pass through all layers
    Tensor *hidden1 = fc1->forward(fc1, input);
    Tensor *hidden2 = fc2->forward(fc2, hidden1);
    Tensor *hidden3 = fc3->forward(fc3, hidden2);
    Tensor *output = fc4->forward(fc4, hidden3);

    // Create an optimizer
    Optimizer *sgd = sgd_create(0.01);

    // Backward pass
    for (int i = 0; i < output->size; i++) 
    {
        output->grad[i] = 1.0;
    }
    output->backward(output, output->grad);

    // Update params
    sgd->step(sgd, model->params, model->num_params);

    // Print tensors
    print_tensor(output, "output");
    print_tensor(fc4->weights, fc4->weights->name);
    print_tensor(fc4->bias, fc4->bias->name);
    print_tensor(fc3->weights, fc3->weights->name);
    print_tensor(fc3->bias, fc3->bias->name);
    print_tensor(fc2->weights, fc2->weights->name);
    print_tensor(fc2->bias, fc2->bias->name);
    print_tensor(fc1->weights, fc1->weights->name);
    print_tensor(fc1->bias, fc1->bias->name);

    // Save the model
    model_save(model, "model.bin");

    // Free memory
    tensor_free(input);
    optimizer_free(sgd);
    model_free(model);

    return 0;
}