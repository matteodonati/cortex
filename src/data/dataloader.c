#include <stdlib.h>
#include <string.h>
#include "data/dataloader.h"

DataLoader* dataloader_create(Dataset *dataset, int batch_size, bool shuffle) 
{
    DataLoader *loader = (DataLoader *)malloc(sizeof(DataLoader));
    loader->dataset = dataset;
    loader->batch_size = batch_size;
    loader->num_batches = dataset->num_samples / batch_size;
    loader->indices = (int *)malloc(dataset->num_samples * sizeof(int));
    for (int i = 0; i < dataset->num_samples; i++) 
    {
        loader->indices[i] = i;
    }
    loader->shuffle = shuffle;
    loader->current_batch = 0;

    if (shuffle) 
    {
        dataloader_shuffle(loader);
    }

    return loader;
}

void dataloader_shuffle(DataLoader *loader) 
{
    srand(time(NULL));
    for (int i = loader->dataset->num_samples - 1; i > 0; i--) 
    {
        int j = rand() % (i + 1);
        int temp = loader->indices[i];
        loader->indices[i] = loader->indices[j];
        loader->indices[j] = temp;
    }
}

void dataloader_get_batch(DataLoader *loader, Tensor **xs, Tensor **ys) 
{
    int start_idx = loader->current_batch * loader->batch_size;
    int remaining_samples = loader->dataset->num_samples - start_idx;
    int current_batch_size = remaining_samples < loader->batch_size ? remaining_samples : loader->batch_size;

    // Determine the total size of each sample in the batch
    int x_total_size = 1;
    int y_total_size = 1;
    for (int i = 0; i < loader->dataset->x_ndim; i++)
    {
        x_total_size *= loader->dataset->x_shape[i];
    }
    for (int i = 0; i < loader->dataset->y_ndim; i++)
    {
        y_total_size *= loader->dataset->y_shape[i];
    }

    // Create tensors for the batch, with appropriate shapes
    int *x_batch_shape = (int *)malloc((loader->dataset->x_ndim + 1) * sizeof(int));
    int *y_batch_shape = (int *)malloc((loader->dataset->y_ndim + 1) * sizeof(int));

    x_batch_shape[0] = current_batch_size;
    y_batch_shape[0] = current_batch_size;
    memcpy(x_batch_shape + 1, loader->dataset->x_shape, loader->dataset->x_ndim * sizeof(int));
    memcpy(y_batch_shape + 1, loader->dataset->y_shape, loader->dataset->y_ndim * sizeof(int));

    Tensor *tmp_xs = tensor_zeros(NULL, x_batch_shape, loader->dataset->x_ndim + 1);
    Tensor *tmp_ys = tensor_zeros(NULL, y_batch_shape, loader->dataset->y_ndim + 1);

    free(x_batch_shape);
    free(y_batch_shape);

    // Fill the batch tensors with the data
    for (int i = 0; i < current_batch_size; i++) 
    {
        Tensor *x; 
        Tensor *y;
        dataset_get_sample(loader->dataset, loader->indices[start_idx + i], &x, &y);
        tensor_set_slice(tmp_xs, x, i);
        tensor_set_slice(tmp_ys, y, i);
        tensor_free(x);
        tensor_free(y);
    }

    *xs = tmp_xs;
    *ys = tmp_ys;

    loader->current_batch++;
    if (loader->current_batch >= loader->num_batches) 
    {
        loader->current_batch = 0;
        if (loader->shuffle) 
        {
            dataloader_shuffle(loader);
        }
    }
}

void dataloader_free(DataLoader *loader) 
{
    if (loader) 
    {
        free(loader->indices);
        free(loader);
    }
}