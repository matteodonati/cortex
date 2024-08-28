#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cortex.h>

void print_tensor(Tensor *tensor, const char *name) 
{
    printf("Tensor %s:\n", name);
    printf("shape:\n");
    print_tensor_shape(tensor);
    printf("data:\n");
    print_tensor_data(tensor);
    printf("grad:\n");
    print_tensor_grad(tensor);
    printf("\n");
}

void generate_regression_data(float *x_data, float *y_data, int num_samples) 
{
    for (int i = 0; i < num_samples; i++) 
    {
        x_data[i] = (float)i * 0.1;
        y_data[i] = sinf(x_data[i]);
    }
}

void generate_classification_data(float *x_data, float *y_data, int num_samples, int num_classes) 
{
    for (int i = 0; i < num_samples; i++) 
    {
        x_data[i * 2] = (float)(rand() % 100) / 100.0f;
        x_data[i * 2 + 1] = (float)(rand() % 100) / 100.0f;
        y_data[i] = (x_data[i * 2] + x_data[i * 2 + 1] > 1.0f) ? 1 : 0;
    }
}

void regression()
{
    // Generate sine wave data
    int num_samples = 100;
    int batch_size = 10;
    int num_batches = num_samples / batch_size;

    float *x_data = (float *)malloc(num_samples * sizeof(float));
    float *y_data = (float *)malloc(num_samples * sizeof(float));
    generate_regression_data(x_data, y_data, num_samples);

    // Layers
    Layer *fc1 = dense_create("fc1", 1, 256);
    Layer *fc2 = dense_create("fc2", 256, 128);
    Layer *fc3 = dense_create("fc3", 128, 1);

    // Model
    int num_layers = 3;
    Layer *layers[] = {fc1, fc2, fc3};
    Model *model = model_create(layers, num_layers);

    // Optimizer
    Optimizer *sgd = sgd_create(0.01);

    // Measure time
    clock_t start_time = clock();

    // Train
    int num_epochs = 10;
    for (int epoch = 0; epoch < num_epochs; epoch++) 
    {
        float epoch_loss = 0.0f;

        int batch;
        progress(batch, num_batches)
        {
            // Prepare the mini-batch
            int x_shape[] = {batch_size, 1};
            int y_shape[] = {batch_size, 1};

            Tensor *x_batch = tensor_from_array("x_batch", &x_data[batch * batch_size], x_shape, 2);
            Tensor *y_batch = tensor_from_array("y_batch", &y_data[batch * batch_size], y_shape, 2);

            // Forward pass
            Tensor *x1 = relu_f(forward(fc1, x_batch));
            Tensor *x2 = relu_f(forward(fc2, x1));
            Tensor *y_pred = forward(fc3, x2);

            // Calculate the loss using MSE
            Tensor *loss = mse_loss(y_batch, y_pred);

            // Accumulate the loss
            epoch_loss += loss->data[0];
            
            // Backward pass
            backward(loss);

            // Update parameters
            optimizer_step(sgd, model->params, model->num_params);

            // Reset gradients
            model_zero_grad(model);

            // Free temporary tensors
            for (int i = 0; i < num_layers; i++) 
            {
                for (int j = 0; j < layers[i]->tensor_count; j++) 
                {
                    tensor_free(layers[i]->tensors[j]);
                }
                free(layers[i]->tensors);
            }
            tensor_free(x_batch);
            tensor_free(y_batch);
            tensor_free(x1);
            tensor_free(x2);
            tensor_free(loss);
        }

        // Print loss per epoch
        printf("Epoch %03d - loss: %f\n", epoch + 1, epoch_loss / num_batches);
    }

    clock_t end_time = clock();
    double training_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Training time: %f seconds\n", training_time);

    // Free memory
    optimizer_free(sgd);
    model_free(model);
    free(x_data);
    free(y_data);
}

void classification()
{
    // Generate synthetic classification data
    int num_samples = 100;
    int num_classes = 2;
    int input_dim = 2;
    int batch_size = 10;
    int num_batches = num_samples / batch_size;

    float *x_data = (float *)malloc(num_samples * input_dim * sizeof(float));
    float *y_data = (float *)malloc(num_samples * sizeof(float));
    generate_classification_data(x_data, y_data, num_samples, num_classes);

    // Layers
    Layer *fc1 = dense_create("fc1", input_dim, 128);
    Layer *fc2 = dense_create("fc2", 128, 64);
    Layer *fc3 = dense_create("fc3", 64, num_classes);

    // Model
    int num_layers = 3;
    Layer *layers[] = {fc1, fc2, fc3};
    Model *model = model_create(layers, num_layers);

    // Optimizer
    Optimizer *sgd = sgd_create(0.01);

    // Measure time
    clock_t start_time = clock();

    // Train
    int num_epochs = 10;
    for (int epoch = 0; epoch < num_epochs; epoch++) 
    {
        float epoch_loss = 0.0f;

        int batch;
        progress(batch, num_batches)
        {
            // Prepare the mini-batch
            int x_shape[] = {batch_size, input_dim};
            int y_shape[] = {batch_size};

            Tensor *x_batch = tensor_from_array("x_batch", &x_data[batch * batch_size * input_dim], x_shape, 2);
            Tensor *y_batch = tensor_from_array("y_batch", &y_data[batch * batch_size], y_shape, 1);

            // Forward pass
            Tensor *x1 = relu_f(forward(fc1, x_batch));
            Tensor *x2 = relu_f(forward(fc2, x1));
            Tensor *y_pred = softmax_f(forward(fc3, x2), 1);

            // Calculate the loss using cross-entropy
            Tensor *loss = cross_entropy_loss(y_batch, y_pred);

            // Accumulate the loss
            epoch_loss += loss->data[0];
            
            // Backward pass
            backward(loss);

            // Update parameters
            optimizer_step(sgd, model->params, model->num_params);

            // Reset gradients
            model_zero_grad(model);

            // Free temporary tensors
            for (int i = 0; i < num_layers; i++) 
            {
                for (int j = 0; j < layers[i]->tensor_count; j++) 
                {
                    tensor_free(layers[i]->tensors[j]);
                }
                free(layers[i]->tensors);
            }
            tensor_free(x_batch);
            tensor_free(y_batch);
            tensor_free(x1);
            tensor_free(x2);
            tensor_free(y_pred);
            tensor_free(loss);
        }

        // Print loss per epoch
        printf("Epoch %03d - loss: %f\n", epoch + 1, epoch_loss / num_batches);
    }

    clock_t end_time = clock();
    double training_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Training time: %f seconds\n", training_time);

    // Free memory
    optimizer_free(sgd);
    model_free(model);
    free(x_data);
    free(y_data);
}

int main() 
{
    srand(time(NULL));
    regression();
    classification();
    return 0;
}