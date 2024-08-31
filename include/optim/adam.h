#ifndef ADAM_H
#define ADAM_H

#include "optim/optim.h"

typedef struct Adam
{
    Optimizer base;
    int t;
    float beta1;
    float beta2;
    float epsilon;
    int num_params;
    Tensor **m;
    Tensor **v;
} Adam;

Optimizer* adam_create(float learning_rate, float beta1, float beta2, float epsilon);
void adam_step(Optimizer *self, Tensor **params, int num_params);
void adam_free(Optimizer *self);

#endif