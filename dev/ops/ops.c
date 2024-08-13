#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "tensor/tensor.h"
#include "tensor/utils/utils.h"
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"

void print_tensor(Tensor *tensor, const char *name) {
    printf("Tensor %s:\n", name);
    print_tensor_shape(tensor);
    print_tensor_data(tensor);
    printf("\n");
}

void tensor_ops()
{
    // Test tensor creation functions
    int shape_a[] = {3, 3};
    int shape_b[] = {3, 3};
    Tensor *a = tensor_rand(shape_a, 2);
    Tensor *b = tensor_rand(shape_b, 2);

    // Print initial tensors
    print_tensor(a, "a");
    print_tensor(b, "b");

    // Test binary operations
    Tensor *sum = tensor_add(a, b);
    Tensor *sub = tensor_sub(a, b);
    Tensor *mul = tensor_mul(a, b);
    Tensor *div = tensor_div(a, b);

    // Print results of binary operations
    print_tensor(sum, "a + b");
    print_tensor(sub, "a - b");
    print_tensor(mul, "a * b");
    print_tensor(div, "a / b");

    // Test unary operations
    Tensor *neg = tensor_negate(a);
    Tensor *abs = tensor_abs(a);
    Tensor *sqrt = tensor_sqrt(a);
    Tensor *exp = tensor_exp(a);

    // Print results of unary operations
    print_tensor(neg, "-a");
    print_tensor(abs, "abs(a)");
    print_tensor(sqrt, "sqrt(a)");
    print_tensor(exp, "exp(a)");

    // Test matrix multiplication (dot product)
    Tensor *dot = tensor_dot(a, b);
    print_tensor(dot, "a . b");

    // Test transpose
    // int axes[] = {1, 0};
    // Tensor *trans = tensor_transpose(a, axes);
    // print_tensor(trans, "transpose(a)");

    // Test reshape
    int new_shape[] = {1, 9};
    Tensor *reshaped = tensor_reshape(a, new_shape, 2);
    print_tensor(reshaped, "reshaped(a)");

    // Clean up memory
    tensor_free(a);
    tensor_free(b);
    tensor_free(sum);
    tensor_free(sub);
    tensor_free(mul);
    tensor_free(div);
    tensor_free(neg);
    tensor_free(abs);
    tensor_free(sqrt);
    tensor_free(exp);
    tensor_free(dot);
    //tensor_free(trans);
    tensor_free(reshaped);
}

int main() 
{
    srand(time(NULL));
    tensor_ops();
    return 0;
}
