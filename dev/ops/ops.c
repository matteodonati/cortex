#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cortex.h>

void print_tensor(Tensor *tensor, const char *name) {
    printf("Tensor %s:\n", name);
    printf("shape:\n");
    print_tensor_shape(tensor);
    printf("data:\n");
    print_tensor_data(tensor);
    printf("grad:\n");
    print_tensor_grad(tensor);
    printf("\n");
}

void tensor_ops()
{
    int ndim_a = 4;
    int ndim_b = 4;
    int shape_a[] = {2, 3, 4, 6};
    int shape_b[] = {2, 3, 6, 5};
    int shape_c[] = {2, 3, 4, 5};

    Tensor *a = tensor_full(shape_a, ndim_a, 3.0);
    Tensor *b = tensor_full(shape_b, ndim_b, 4.0);

    // Forward pass
    Tensor *c = tensor_matmul(a, b);  // c = a @ b -> Shape: (2, 3, 4, 5)
    Tensor *d = tensor_add(c, c);  // d = c + c -> Shape: (2, 3, 4, 5)
    Tensor *e = tensor_add(d, c);  // e = d + c -> Shape: (2, 3, 4, 5)
    Tensor *f = tensor_div(e, c);  // f = e / c -> Shape: (2, 3, 4, 5)
    
    // Backward pass
    for (int i = 0; i < f->size; i++)
    {
        f->grad[i] = 1.0;
    }
    f->backward(f, f->grad);

    // Print tensors
    print_tensor(a, "a");
    print_tensor(b, "b");
    print_tensor(c, "c");
    print_tensor(d, "d");
    print_tensor(e, "e");
    print_tensor(f, "f");

    // Free tensors
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(e);
    tensor_free(f);
}

int main() 
{
    srand(time(NULL));
    tensor_ops();
    return 0;
}
