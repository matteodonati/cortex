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
    int shape_a[] = {2, 3, 4, 6};

    Tensor *a = tensor_full(shape_a, ndim_a, 3.0);
    Tensor *b = tensor_full(shape_a, ndim_a, 4.0);  // Shape matches a

    // Unary operations
    Tensor *neg = tensor_negate(a);
    Tensor *abs = tensor_abs(a);
    Tensor *sqrt = tensor_sqrt(a);
    Tensor *exp = tensor_exp(a);
    Tensor *log = tensor_log(a);

    // Binary operations
    Tensor *add = tensor_add(a, b);
    Tensor *sub = tensor_sub(a, b);
    Tensor *mul = tensor_mul(a, b);
    Tensor *div = tensor_div(a, b);

    // Additional operations (reshape, transpose, max, min, argmax, argmin)
    Tensor *reshaped = tensor_reshape(a, (int []){2, 3, 6, 4}, 4);
    Tensor *transposed = tensor_transpose(a, (int []){0, 2, 1, 3});
    Tensor *max_result = tensor_max(a, 2);
    Tensor *min_result = tensor_min(a, 2);
    Tensor *argmax_result = tensor_argmax(a, 2);
    Tensor *argmin_result = tensor_argmin(a, 2);

    // Initialize gradients for backward pass
    for (int i = 0; i < a->size; i++) 
    {
        neg->grad[i] = 1.0;
        abs->grad[i] = 1.0;
        sqrt->grad[i] = 1.0;
        exp->grad[i] = 1.0;
        log->grad[i] = 1.0;
        add->grad[i] = 1.0;
        sub->grad[i] = 1.0;
        mul->grad[i] = 1.0;
        div->grad[i] = 1.0;
    }

    for (int i = 0; i < reshaped->size; i++) 
    {
        reshaped->grad[i] = 1.0;
    }

    for (int i = 0; i < transposed->size; i++) 
    {
        transposed->grad[i] = 1.0;
    }

    for (int i = 0; i < max_result->size; i++) 
    {
        max_result->grad[i] = 1.0;
    }

    for (int i = 0; i < min_result->size; i++) 
    {
        min_result->grad[i] = 1.0;
    }

    // Perform backward passes
    neg->backward(neg, neg->grad);
    abs->backward(abs, abs->grad);
    sqrt->backward(sqrt, sqrt->grad);
    exp->backward(exp, exp->grad);
    log->backward(log, log->grad);

    add->backward(add, add->grad);
    sub->backward(sub, sub->grad);
    mul->backward(mul, mul->grad);
    div->backward(div, div->grad);

    reshaped->backward(reshaped, reshaped->grad);
    transposed->backward(transposed, transposed->grad);
    max_result->backward(max_result, max_result->grad);
    min_result->backward(min_result, min_result->grad);

    // Print all the results
    print_tensor(a, "a");
    print_tensor(b, "b");
    print_tensor(neg, "neg");
    print_tensor(abs, "abs");
    print_tensor(sqrt, "sqrt");
    print_tensor(exp, "exp");
    print_tensor(log, "log");
    print_tensor(add, "add");
    print_tensor(sub, "sub");
    print_tensor(mul, "mul");
    print_tensor(div, "div");
    print_tensor(reshaped, "reshaped");
    print_tensor(transposed, "transposed");
    print_tensor(max_result, "max");
    print_tensor(min_result, "min");
    print_tensor(argmax_result, "argmax");
    print_tensor(argmin_result, "argmin");

    // Free all tensors
    tensor_free(a);
    tensor_free(b);
    tensor_free(neg);
    tensor_free(abs);
    tensor_free(sqrt);
    tensor_free(exp);
    tensor_free(log);
    tensor_free(add);
    tensor_free(sub);
    tensor_free(mul);
    tensor_free(div);
    tensor_free(reshaped);
    tensor_free(transposed);
    tensor_free(max_result);
    tensor_free(min_result);
    tensor_free(argmax_result);
    tensor_free(argmin_result);
}

void tensor_matmul_op() 
{
    int ndim_a = 4;
    int ndim_b = 4;
    int shape_a[] = {2, 3, 4, 6};
    int shape_b[] = {2, 3, 6, 5};

    Tensor *a = tensor_full(shape_a, ndim_a, 3.0);
    Tensor *b = tensor_full(shape_b, ndim_b, 4.0);

    // Matmul operation
    Tensor *matmul = tensor_matmul(a, b);

    // Initialize gradients for backward pass
    for (int i = 0; i < matmul->size; i++) 
    {
        matmul->grad[i] = 1.0;
    }

    // Perform backward pass
    matmul->backward(matmul, matmul->grad);

    // Print the results
    print_tensor(a, "a");
    print_tensor(b, "b");
    print_tensor(matmul, "matmul");

    // Free tensors
    tensor_free(a);
    tensor_free(b);
    tensor_free(matmul);
}

int main() {
    srand(time(NULL));
    tensor_ops();
    tensor_matmul_op();
    return 0;
}