#include <stdlib.h>
#include "optim/optim.h"

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