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

void dataloader_free(DataLoader *loader) 
{
    if (loader) 
    {
        free(loader->indices);
        free(loader);
    }
}

void dataloader_shuffle(DataLoader *loader) 
{
    for (int i = loader->dataset->num_samples - 1; i > 0; i--) 
    {
        int j = rand() % (i + 1);
        int tmp = loader->indices[i];
        loader->indices[i] = loader->indices[j];
        loader->indices[j] = tmp;
    }
}

void dataloader_get_batch(DataLoader *loader, Tensor **xs, Tensor **ys) 
{
    int start_idx = loader->current_batch * loader->batch_size;
    int end_idx = start_idx + loader->batch_size;

    Tensor *tmp_xs = tensor_zeros(NULL, (int[]){loader->batch_size, loader->dataset->num_features}, 2);
    Tensor *tmp_ys = tensor_zeros(NULL, (int[]){loader->batch_size, loader->dataset->num_outputs}, 2);

    for (int i = start_idx; i < end_idx; i++) 
    {
        Tensor *x; 
        Tensor *y;
        dataset_get_sample(loader->dataset, loader->indices[i], &x, &y);
        tensor_set_slice(tmp_xs, x, i - start_idx);
        tensor_set_slice(tmp_ys, y, i - start_idx);
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