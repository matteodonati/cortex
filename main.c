#include <stdio.h>
#include <cortex.h>

int main()
{
    int shape[2] = {2, 3};

    /* Create tensor */
    Tensor *tensor = create(shape, 2);

    /* Initialize tensor */
    for(int i = 0; i < tensor->size; i++) 
    {
        tensor->data[i] = (float)i;
    }

    /* Print tensor */
    printf("Tensor data:\n");
    for(int i = 0; i < tensor->ndim; i++) 
    {
        printf("Dimension %d size: %d\n", i, tensor->shape[i]);
        printf("Dimension %d stride: %d\n", i, tensor->stride[i]);
    }
    for(int i = 0; i < tensor->size; i++) 
    {
        printf("%f ", tensor->data[i]);
    }
    printf("\n");

    /* Free tensor */
    free_tensor(tensor);

    return 0;
}
