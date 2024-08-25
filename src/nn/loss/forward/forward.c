#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ops/utils/utils.h"
#include "nn/loss/forward/forward.h"
#include "nn/loss/backward/backward.h"

Tensor* mse_loss(Tensor *y_true, Tensor *y_pred) 
{
    if (y_true->size != y_pred->size) 
    {
        fprintf(stderr, "Error: y_true and y_pred must have the same size in MSE loss.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *loss = tensor_zeros("mse_loss", (int[]){1}, 1);
    for (int i = 0; i < y_pred->size; i++) 
    {
        float diff = y_pred->data[i] - y_true->data[i];
        loss->data[0] += diff * diff;
    }

    loss->data[0] /= y_pred->size;
    loss->grad[0] = 1.0;
    loss->backward = &mse_loss_backward;
    loss->grad_a = y_true;
    loss->grad_b = y_pred;

    return loss;
}

Tensor* cross_entropy_loss(Tensor *y_true, Tensor *y_pred) 
{
    if (y_true == NULL || y_pred == NULL) 
    {
        fprintf(stderr, "Error: One of the input tensors is NULL in cross_entropy_loss.\n");
        exit(EXIT_FAILURE);
    }

    // One-hot encode
    int num_classes = y_pred->shape[y_pred->ndim - 1];
    Tensor *y_true_one_hot = one_hot_encode(y_true, num_classes);

    Tensor *loss = tensor_zeros("cross_entropy_loss", (int[]){1}, 1);
    for (int i = 0; i < y_true_one_hot->size; i++) 
    {
        loss->data[0] -= y_true_one_hot->data[i] * logf(y_pred->data[i]);
    }
    tensor_free(y_true_one_hot);

    loss->data[0] /= y_true->size;
    loss->grad[0] = 1.0;
    loss->backward = &cross_entropy_loss_backward;
    loss->grad_a = y_true;
    loss->grad_b = y_pred;

    return loss;
}