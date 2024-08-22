#include <stdlib.h>
#include "nn/parameters/parameters.h"

void parameters_free(Parameters *params)
{
    if (params->free)
    {
        params->free(params);
    }
    else
    {
        free(params);
    }
}