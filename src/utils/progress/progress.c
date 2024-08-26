#include "utils/progress/progress.h"

void progress_bar_update(ProgressBar *pb, int value) 
{
    if (value > pb->total) 
    {
        return;
    }

    int width = 50;
    pb->current = value;
    int pos = (pb->current * width) / pb->total;

    // Clear the line first to avoid any remnants from previous prints
    printf("\r"); 
    fflush(stdout);

    // Reprint the progress bar
    for (int i = 0; i < width; i++) 
    {
        if (i < pos) 
        {
            printf("█");
        } 
        else 
        {
            printf(" ");
        }
    }

    int percent = (pb->current * 100) / pb->total;
    printf(" %3d%% (%d/%d)", percent, pb->current, pb->total);
    fflush(stdout);

    // Print a newline if progress is complete
    if (pb->current == pb->total) 
    {
        printf("\n");
    }
}