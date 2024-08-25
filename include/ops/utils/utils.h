#ifndef OPS_UTILS_H
#define OPS_UTILS_H

#include "tensor/tensor.h"

Tensor* one_hot_encode(Tensor *labels, int num_classes);

#endif