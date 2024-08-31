#include <stdlib.h>
#include <math.h>
#include "optim/adamw.h"

Optimizer* adamw_create(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay) 
{
    AdamW *optimizer = (AdamW *)malloc(sizeof(AdamW));
    optimizer->base.learning_rate = learning_rate;
    optimizer->base.optim_type = OPTIM_TYPE_ADAMW;
    optimizer->base.step = &adamw_step;
    optimizer->base.free = &adamw_free;
    optimizer->t = 0;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->weight_decay = weight_decay;
    optimizer->m = NULL;
    optimizer->v = NULL;
    optimizer->num_params = 0;
    return (Optimizer *)optimizer;
}

void adamw_step(Optimizer *self, Tensor **params, int num_params) 
{
    AdamW *adamw = (AdamW *)self;
    if (adamw->m == NULL) 
    {
        adamw->m = (Tensor **)malloc(num_params * sizeof(Tensor *));
        adamw->v = (Tensor **)malloc(num_params * sizeof(Tensor *));
        adamw->num_params = num_params;

        for (int i = 0; i < num_params; i++) 
        {
            adamw->m[i] = tensor_zeros(NULL, params[i]->shape, params[i]->ndim);
            adamw->v[i] = tensor_zeros(NULL, params[i]->shape, params[i]->ndim);
        }
    }

    adamw->t += 1;
    float lr_t = self->learning_rate * sqrtf(1.0f - powf(adamw->beta2, adamw->t)) / (1.0f - powf(adamw->beta1, adamw->t));

    for (int i = 0; i < num_params; i++) 
    {
        Tensor *param = params[i];
        if (param->frozen) 
        {
            continue;
        }

        for (int j = 0; j < param->size; j++) 
        {
            // Update biased first moment estimate
            adamw->m[i]->data[j] = adamw->beta1 * adamw->m[i]->data[j] + (1.0f - adamw->beta1) * param->grad[j];
            
            // Update biased second raw moment estimate
            adamw->v[i]->data[j] = adamw->beta2 * adamw->v[i]->data[j] + (1.0f - adamw->beta2) * param->grad[j] * param->grad[j];
            
            // Compute the corrected first moment estimate
            float m_hat = adamw->m[i]->data[j] / (1.0f - powf(adamw->beta1, adamw->t));
            
            // Compute the corrected second moment estimate
            float v_hat = adamw->v[i]->data[j] / (1.0f - powf(adamw->beta2, adamw->t));
            
            // Update parameters with decoupled weight decay
            param->data[j] -= lr_t * (m_hat / (sqrtf(v_hat) + adamw->epsilon) + adamw->weight_decay * param->data[j]);
        }
    }
}

void adamw_free(Optimizer *self)
{
    AdamW *adamw = (AdamW *)self;
    if (adamw->m != NULL) 
    {
        for (int i = 0; i < adamw->num_params; i++) 
        {
            tensor_free(adamw->m[i]);
            tensor_free(adamw->v[i]);
        }
        free(adamw->m);
        free(adamw->v);
    }
    free(adamw);
}