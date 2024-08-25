#ifndef NN_LOSS_FORWARD_H
#define NN_LOSS_FORWARD_H

#include "tensor/tensor.h"

Tensor* mse_loss(Tensor *y_true, Tensor *y_pred);
Tensor* cross_entropy_loss(Tensor *y_true, Tensor *y_pred);

#endif