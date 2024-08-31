#ifndef ADAMW_H
#define ADAMW_H

#include "optim/optim.h"

typedef struct AdamW 
{
    Optimizer base;
    int t;
    float beta1;
    float beta2;
    float epsilon;
    int num_params;
    float weight_decay;
    Tensor **m;
    Tensor **v;
} AdamW;

Optimizer* adamw_create(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay);
void adamw_step(Optimizer *self, Tensor **params, int num_params);
void adamw_free(Optimizer *self);

#endif