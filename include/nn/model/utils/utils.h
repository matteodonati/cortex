#ifndef NN_MODEL_UTILS_H
#define NN_MODEL_UTILS_H

#include "nn/model/model.h"

void model_save(Model *model, const char *filename);
void model_load(Model *model, const char *filename);

#endif