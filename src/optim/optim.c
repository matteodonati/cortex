#include <stdlib.h>
#include "optim/optim.h"

void optimizer_step(Optimizer *optim, Tensor **params, int num_params)
{
    if (optim->step)
    {
        optim->step(optim, params, num_params);
    }
}

void optimizer_free(Optimizer *optimizer) 
{
    if (optimizer) 
    {
        if (optimizer->free)
        {
            optimizer->free(optimizer);
        }
        else
        {
            free(optimizer);
        }
    }
}