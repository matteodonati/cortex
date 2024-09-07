#include <math.h>
#include <stdlib.h>
#include "optim/adam.h"

Optimizer* adam_create(float learning_rate, float beta1, float beta2, float epsilon)
{
    Adam *optimizer = (Adam *)malloc(sizeof(Adam));
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.optim_type = OPTIM_TYPE_ADAM;
    optimizer->base.step = &adam_step;
    optimizer->base.free = &adam_free;
    optimizer->m = NULL;
    optimizer->v = NULL;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->num_params = 0;
    optimizer->t = 0;    
    return (Optimizer *)optimizer;
}

void adam_step(Optimizer *self, Tensor **params, int num_params)
{
    Adam *adam = (Adam *)self;
    adam->t += 1;

    if (adam->m == NULL) 
    {
        adam->m = (Tensor **)malloc(num_params * sizeof(Tensor *));
        adam->v = (Tensor **)malloc(num_params * sizeof(Tensor *));
        adam->num_params = num_params;

        for (int i = 0; i < num_params; i++) 
        {
            adam->m[i] = tensor_zeros(NULL, params[i]->shape, params[i]->ndim);
            adam->v[i] = tensor_zeros(NULL, params[i]->shape, params[i]->ndim);
        }
    }

    for (int i = 0; i < num_params; i++) 
    {
        Tensor *param = params[i];
        if (param->frozen) 
        {
            continue;
        }

        Tensor *m = adam->m[i];
        Tensor *v = adam->v[i];

        for (int j = 0; j < param->size; j++) 
        {
            // Update biased first moment estimate
            m->data[j] = adam->beta1 * m->data[j] + (1.0f - adam->beta1) * param->grad[j];

            // Update biased second moment estimate
            v->data[j] = adam->beta2 * v->data[j] + (1.0f - adam->beta2) * param->grad[j] * param->grad[j];

            // Compute bias-corrected first moment estimate
            float m_hat = m->data[j] / (1.0f - powf(adam->beta1, adam->t));

            // Compute bias-corrected second moment estimate
            float v_hat = v->data[j] / (1.0f - powf(adam->beta2, adam->t));

            // Update parameters
            param->data[j] -= self->learning_rate * m_hat / (sqrtf(v_hat) + adam->epsilon);
        }
    }
}

void adam_free(Optimizer *self)
{
    Adam *adam = (Adam *)self;
    if (adam->m != NULL) 
    {
        for (int i = 0; i < adam->num_params; i++) 
        {
            tensor_free(adam->m[i]);
            tensor_free(adam->v[i]);
        }
        free(adam->m);
        free(adam->v);
    }
    free(adam);
}