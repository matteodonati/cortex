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
    float x_data[] = {1.5, -2.3, 3.7, -4.1, 5.2, -6.8, 7.9, -8.4};
    int x_shape[] = {2, 4};
    Tensor *x = tensor_from_array("x", x_data, x_shape, 2);

    // First Dense layer, shape {4, 3}
    Layer *fc1 = dense_create("fc1", 4, 3);
    DenseParameters *fc1_params = (DenseParameters *)(fc1->params);
    float w1[] = {0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 1.4, -1.5};
    float b1[] = {0.2, -0.3, 0.4};
    memcpy(fc1_params->weights->data, w1, sizeof(w1));
    memcpy(fc1_params->bias->data, b1, sizeof(b1));

    // Second Dense layer, shape {3, 2}
    Layer *fc2 = dense_create("fc2", 3, 2);
    DenseParameters *fc2_params = (DenseParameters *)(fc2->params);
    float w2[] = {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6};
    float b2[] = {-0.1, 0.1};
    memcpy(fc2_params->weights->data, w2, sizeof(w2));
    memcpy(fc2_params->bias->data, b2, sizeof(b2));

    // Create the model and add layers
    Layer *layers[] = {fc1, fc2};
    Model *model = model_create(layers, 2);

    // Forward pass through all layers
    Tensor *x1 = fc1->forward(fc1, x);
    Tensor *x2 = fc2->forward(fc2, x1);
    Tensor *y = softmax_f(x2, 1);

    // Create an optimizer
    Optimizer *sgd = sgd_create(0.01);

    // Backward pass
    for (int i = 0; i < y->size; i++) 
    {
        y->grad[i] = 1.0;
    }
    y->backward(y, y->grad);

    // Update params
    sgd->step(sgd, model->params, model->num_params);

    // Print tensors
    print_tensor(y, "y");
    print_tensor(fc2_params->weights, fc2_params->weights->name);
    print_tensor(fc2_params->bias, fc2_params->bias->name);
    print_tensor(fc1_params->weights, fc1_params->weights->name);
    print_tensor(fc1_params->bias, fc1_params->bias->name);

    // Save the model
    model_save(model, "model.bin");

    // Free memory
    tensor_free(x);
    tensor_free(y);
    optimizer_free(sgd);
    model_free(model);

    return 0;
}