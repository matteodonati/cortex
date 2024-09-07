#ifndef ADAM_H
#define ADAM_H

#include "optim/optim.h"

typedef struct Adam
{
    Optimizer base;
    Tensor **m;
    Tensor **v;
    float beta1;
    float beta2;
    float epsilon;
    int num_params;
    int t;
} Adam;

Optimizer* adam_create(float learning_rate, float beta1, float beta2, float epsilon);
void adam_step(Optimizer *self, Tensor **params, int num_params);
void adam_free(Optimizer *self);

#endif