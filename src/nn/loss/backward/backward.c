#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ops/utils/utils.h"
#include "nn/loss/backward/backward.h"

void mse_loss_backward(Tensor *loss) 
{
    Tensor *y_true = loss->grad_a;
    Tensor *y_pred = loss->grad_b;

    for (int i = 0; i < y_pred->size; i++) 
    {
        y_pred->grad[i] += (2.0 / y_pred->size) * (y_pred->data[i] - y_true->data[i]) * loss->grad[0];
    }

    backward(y_pred);
}

void cross_entropy_loss_backward(Tensor *loss) 
{
    Tensor *y_true = loss->grad_a;
    Tensor *y_pred = loss->grad_b;
    int batch_size = y_true->shape[0];
    int num_classes = y_pred->shape[1];

    for (int i = 0; i < batch_size; i++) 
    {
        int class_index = (int)y_true->data[i];
        int idx = i * num_classes + class_index;
        y_pred->grad[idx] += -1.0 / (y_pred->data[idx] + 1e-9) * loss->grad[0];
        y_pred->grad[idx] /= batch_size;
    }

    backward(y_pred);
}