#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cortex.h>

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
    int batch_size = 32;

    float *x_data = (float *)malloc(num_samples * sizeof(float));
    float *y_data = (float *)malloc(num_samples * sizeof(float));
    generate_regression_data(x_data, y_data, num_samples);

    // Create Dataset and DataLoader
    int x_shape[] = {1};  // Shape of a single input sample
    int y_shape[] = {1};  // Shape of a single output sample
    Dataset *dataset = dataset_create(x_data, y_data, num_samples, x_shape, 1, y_shape, 1);
    DataLoader *dataloader = dataloader_create(dataset, batch_size, true);

    // Layers
    Layer *fc1 = dense_create("fc1", 1, 32);
    Layer *fc2 = dense_create("fc2", 32, 16);
    Layer *fc3 = dense_create("fc3", 16, 1);

    // Model
    int num_layers = 3;
    Layer *layers[] = {fc1, fc2, fc3};
    Model *model = model_create(layers, num_layers);

    // Optimizer
    Optimizer *optim = adamw_create(0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

    // Measure time
    clock_t start_time = clock();

    // Train
    int num_epochs = 100;
    for (int epoch = 0; epoch < num_epochs; epoch++) 
    {
        float epoch_loss = 0.0f;
        
        int batch;
        progress (batch, dataloader->num_batches)
        {
            Tensor *x_batch;
            Tensor *y_batch;
            dataloader_get_batch(dataloader, &x_batch, &y_batch);

            // Forward pass
            Tensor *x1 = relu_f(layer_forward(fc1, x_batch));
            Tensor *x2 = relu_f(layer_forward(fc2, x1));
            Tensor *y_pred = layer_forward(fc3, x2);

            // Calculate the loss using MSE
            Tensor *loss = mse_loss(y_batch, y_pred);

            // Accumulate the loss
            epoch_loss += loss->data[0];
            
            // Backward pass
            backward(loss);

            // Update parameters
            optimizer_step(optim, model->params, model->num_params);

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
        printf("Epoch %03d - loss: %f\n", epoch + 1, epoch_loss / dataloader->num_batches);
    }

    clock_t end_time = clock();
    double training_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("\nTraining time: %f seconds\n", training_time);

    // Free memory
    optimizer_free(optim);
    model_free(model);
    dataloader_free(dataloader);
    dataset_free(dataset);
    free(x_data);
    free(y_data);
}

void classification()
{
    // Generate synthetic classification data
    int num_samples = 100;
    int num_classes = 2;
    int input_dim = 2;
    int batch_size = 32;

    float *x_data = (float *)malloc(num_samples * input_dim * sizeof(float));
    float *y_data = (float *)malloc(num_samples * sizeof(float));
    generate_classification_data(x_data, y_data, num_samples, num_classes);

    // Create Dataset and DataLoader
    int x_shape[] = {input_dim};  // Shape of a single input sample
    int y_shape[] = {1};          // Shape of a single output label
    Dataset *dataset = dataset_create(x_data, y_data, num_samples, x_shape, 1, y_shape, 1);
    DataLoader *dataloader = dataloader_create(dataset, batch_size, true);

    // Layers
    Layer *fc1 = dense_create("fc1", input_dim, 128);
    Layer *fc2 = dense_create("fc2", 128, 64);
    Layer *fc3 = dense_create("fc3", 64, num_classes);

    // Model
    int num_layers = 3;
    Layer *layers[] = {fc1, fc2, fc3};
    Model *model = model_create(layers, num_layers);

    // Optimizer
    Optimizer *optim = adamw_create(0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

    // Measure time
    clock_t start_time = clock();

    // Train
    int num_epochs = 100;
    for (int epoch = 0; epoch < num_epochs; epoch++) 
    {
        float epoch_loss = 0.0f;

        int batch;
        progress (batch, dataloader->num_batches)
        {
            Tensor *x_batch;
            Tensor *y_batch;
            dataloader_get_batch(dataloader, &x_batch, &y_batch);

            // Forward pass
            Tensor *x1 = relu_f(layer_forward(fc1, x_batch));
            Tensor *x2 = relu_f(layer_forward(fc2, x1));
            Tensor *y_pred = softmax_f(layer_forward(fc3, x2), 1);

            // Calculate the loss using cross-entropy
            Tensor *loss = cross_entropy_loss(y_batch, y_pred);

            // Accumulate the loss
            epoch_loss += loss->data[0];
            
            // Backward pass
            backward(loss);

            // Update parameters
            optimizer_step(optim, model->params, model->num_params);

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
        printf("Epoch %03d - loss: %f\n", epoch + 1, epoch_loss / dataloader->num_batches);
    }

    clock_t end_time = clock();
    double training_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("\nTraining time: %f seconds\n", training_time);

    // Free memory
    optimizer_free(optim);
    model_free(model);
    dataloader_free(dataloader);
    dataset_free(dataset);
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