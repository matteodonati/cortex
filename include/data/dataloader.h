#ifndef DATA_DATALOADER_H
#define DATA_DATALOADER_H

#include <stdbool.h>
#include "data/dataset.h"

typedef struct DataLoader
{
    Dataset *dataset;
    int batch_size;
    int num_batches;
    int *indices;
    bool shuffle;
    int current_batch;
} DataLoader;

DataLoader* dataloader_create(Dataset *dataset, int batch_size, bool shuffle);
void dataloader_shuffle(DataLoader *loader);
void dataloader_get_batch(DataLoader *loader, Tensor **batch_data, Tensor **batch_labels);
void dataloader_free(DataLoader *loader);

#endif