#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cortex.h>

void generate_image_classification_data(float *x_data, float *y_data, int num_samples, int height, int width) 
{
    for (int i = 0; i < num_samples; i++) 
    {
        int label = rand() % 2;
        y_data[i] = label;

        for (int c = 0; c < 3; c++) 
        {
            for (int h = 0; h < height; h++) 
            {
                for (int w = 0; w < width; w++) 
                {
                    int idx = ((i * 3 + c) * height + h) * width + w;
                    if (label == 0) 
                    {
                        x_data[idx] = (w < width / 2) ? 1.0f : 0.0f;
                    } 
                    else 
                    {
                        x_data[idx] = (w < width / 2) ? 0.0f : 1.0f;
                    }
                }
            }
        }
    }
}

void convnet_classification()
{
    // Image dimensions and number of classes
    int num_samples = 100;
    int num_classes = 2;
    int height = 28;
    int width = 28;
    int input_channels = 3;
    int batch_size = 32;

    // Generate synthetic image classification data
    float x_data[num_samples * input_channels * height * width];
    float y_data[num_samples];
    generate_image_classification_data(x_data, y_data, num_samples, height, width);

    // Dataset and DataLoader
    int x_shape[] = {input_channels, height, width};  // Shape of a single image
    int y_shape[] = {1};                              // Shape of a single label
    Dataset *dataset = dataset_create(x_data, y_data, num_samples, x_shape, 3, y_shape, 1);
    DataLoader *dataloader = dataloader_create(dataset, batch_size, true);

    // Layers
    Layer *conv1 = conv2d_create("conv1", 3, 16, (int[]){3, 3}, (int[]){1, 1}, (int[]){1, 1});
    Layer *bn1 = batchnorm2d_create("bn1", 16, 1e-5, 0.1f);
    Layer *pool1 = maxpool2d_create("pool1", (int[]){2, 2}, (int[]){2, 2}, (int[]){0, 0});
    Layer *conv2 = conv2d_create("conv2", 16, 32, (int[]){3, 3}, (int[]){1, 1}, (int[]){1, 1});
    Layer *bn2 = batchnorm2d_create("bn2", 32, 1e-5, 0.1f);
    Layer *pool2 = maxpool2d_create("pool2", (int[]){2, 2}, (int[]){2, 2}, (int[]){0, 0});
    Layer *fc1 = dense_create("fc1", 32 * 7 * 7, 128);
    Layer *fc2 = dense_create("fc2", 128, num_classes);

    // Model
    int num_layers = 8;
    Layer *layers[] = {conv1, bn1, pool1, conv2, bn2, pool2, fc1, fc2};
    Model *model = model_create(layers, num_layers);

    // Optimizer
    Optimizer *optim = adamw_create(0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

    // Measure time
    clock_t start_time = clock();

    // Train the model
    model_train(model);
    for (int epoch = 0; epoch < 10; epoch++) 
    {
        float epoch_loss = 0.0f;
        
        int batch;
        progress (batch, dataloader->num_batches)
        {
            Tensor *x_batch;
            Tensor *y_batch;
            dataloader_get_batch(dataloader, &x_batch, &y_batch);

            // Forward pass
            Tensor *x1 = relu_f(layer_forward(conv1, x_batch));
            Tensor *x2 = layer_forward(bn1, x1);
            Tensor *x3 = layer_forward(pool1, x2);
            Tensor *x4 = relu_f(layer_forward(conv2, x3));
            Tensor *x5 = layer_forward(bn2, x4);
            Tensor *x6 = layer_forward(pool2, x5);
            Tensor *x_flat = tensor_reshape(x6, (int []){batch_size, 32 * 7 * 7}, 2);
            Tensor *x7 = relu_f(layer_forward(fc1, x_flat));
            Tensor *y_pred = softmax_f(layer_forward(fc2, x7), 1);

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
            tensor_free(x4);
            tensor_free(x_flat);
            tensor_free(x7);
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
}

int main() 
{
    srand(time(NULL));
    convnet_classification();
    return 0;
}