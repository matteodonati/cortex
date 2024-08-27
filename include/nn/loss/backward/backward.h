#ifndef NN_LOSS_BACKWARD_H
#define NN_LOSS_BACKWARD_H

#include "tensor/tensor.h"

void mse_loss_backward(Tensor *loss);
void cross_entropy_loss_backward(Tensor *loss);

#endif