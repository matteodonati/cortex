#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn/loss/backward/backward.h"

void mse_loss_backward(Tensor *loss, float *grad) 
{
    Tensor *y_true = loss->grad_a;
    Tensor *y_pred = loss->grad_b;

    for (int i = 0; i < y_pred->size; i++) 
    {
        float diff = y_pred->data[i] - y_true->data[i];
        y_pred->grad[i] += (2.0 / y_pred->size) * diff * grad[0];  // Apply the gradient from the loss
    }

    if (y_pred->backward) 
    {
        y_pred->backward(y_pred, y_pred->grad);
    }
}

void cross_entropy_loss_backward(Tensor *loss, float *grad) 
{
    Tensor *y_true = loss->grad_a;
    Tensor *y_pred = loss->grad_b;

    for (int i = 0; i < y_pred->size; i++) 
    {
        if (y_true->data[i] == 1.0) 
        {
            y_pred->grad[i] += -1.0 / (y_pred->data[i] + 1e-9) * grad[0];  // Apply the gradient from the loss
        }
    }

    if (y_pred->backward) 
    {
        y_pred->backward(y_pred, y_pred->grad);
    }
}