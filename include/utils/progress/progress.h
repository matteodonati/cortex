#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct 
{
    int total;
    int current;
} ProgressBar;

#define progress(i, end) \
    ProgressBar pb; \
    pb.total = (end); \
    pb.current = 0; \
    for (i = 0; i < (end); i++, progress_bar_update(&pb, i + 1))

void progress_bar_update(ProgressBar *pb, int value);

#endif