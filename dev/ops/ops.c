#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cortex.h>

void tensor_ops() 
{
    int ndim_a = 4;
    int shape_a[] = {2, 3, 4, 6};

    Tensor *a = tensor_full("a", shape_a, ndim_a, 3.0);
    Tensor *b = tensor_full("b", shape_a, ndim_a, 4.0);

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

    // Additional operations
    Tensor *reshaped = tensor_reshape(a, (int []){2, 3, 6, 4}, 4);
    Tensor *transposed = tensor_transpose(a, (int []){0, 2, 1, 3});
    Tensor *max_result = tensor_max(a, 2);
    Tensor *min_result = tensor_min(a, 2);
    Tensor *argmax_result = tensor_argmax(a, 2);
    Tensor *argmin_result = tensor_argmin(a, 2);
    Tensor *sum_result = tensor_sum(a, (int []){0, 2, 3}, 3);
    Tensor *mean_result = tensor_mean(a, (int []){0, 2, 3}, 3);
    Tensor *var_result = tensor_var(a, (int []){0, 2, 3}, 3, false);
    Tensor *cat_result = tensor_cat(a, b, 3);

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

    for (int i = 0; i < sum_result->size; i++) 
    {
        sum_result->grad[i] = 1.0;
    }

    for (int i = 0; i < mean_result->size; i++) 
    {
        mean_result->grad[i] = 1.0;
    }

    for (int i = 0; i < cat_result->size; i++) 
    {
        cat_result->grad[i] = 1.0;
    }

    for (int i = 0; i < var_result->size; i++) 
    {
        var_result->grad[i] = 1.0;
    }

    // Perform backward passes
    backward(neg);
    backward(abs);
    backward(sqrt);
    backward(exp);
    backward(log);

    backward(add);
    backward(sub);
    backward(mul);
    backward(div);

    backward(reshaped);
    backward(transposed);
    backward(max_result);
    backward(min_result);
    backward(sum_result);
    backward(mean_result);
    backward(var_result);
    backward(cat_result);

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
    print_tensor(sum_result, "sum");
    print_tensor(mean_result, "mean");
    print_tensor(var_result, "var_result");
    print_tensor(cat_result, "cat_result");

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
    tensor_free(sum_result);
    tensor_free(mean_result);
    tensor_free(var_result);
    tensor_free(cat_result);
}

void tensor_matmul_op() 
{
    int ndim_a = 4;
    int ndim_b = 4;
    int shape_a[] = {2, 3, 4, 6};
    int shape_b[] = {2, 3, 6, 5};

    Tensor *a = tensor_full("a", shape_a, ndim_a, 3.0);
    Tensor *b = tensor_full("b", shape_b, ndim_b, 4.0);

    // Matmul operation
    Tensor *matmul = tensor_matmul(a, b);

    // Initialize gradients for backward pass
    for (int i = 0; i < matmul->size; i++) 
    {
        matmul->grad[i] = 1.0;
    }

    // Perform backward pass
    backward(matmul);

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