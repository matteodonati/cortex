#include <stdlib.h>
#include "nn/dense.h"
#include "tensor/tensor.h"
#include "ops/forward/forward.h"
#include "ops/backward/backward.h"
#include "optim/optim.h"

#include <stdio.h>
#include "tensor/utils/utils.h"

Layer* dense_create(int input_dim, int output_dim)
{
    Dense *dense = (Dense *)malloc(sizeof(Dense));
    dense->input_dim = input_dim;
    dense->output_dim = output_dim;
    dense->base.layer_type = LAYER_TYPE_DENSE;
    dense->base.weights = tensor_rand((int[]){input_dim, output_dim}, 2);
    dense->base.bias = tensor_zeros((int[]){output_dim}, 1);
    dense->base.forward = &dense_forward;
    dense->base.backward = &dense_backward;
    dense->base.update_params = &dense_update_params;
    dense->base.free = &dense_free;
    return (Layer *)dense;
}

Tensor* dense_forward(Layer *self, Tensor *input)
{
    self->input = input;
    
    Tensor *z = tensor_matmul(input, self->weights);
    Tensor *output = tensor_add(z, self->bias);

    printf("input: ");
    print_tensor_shape(input);
    print_tensor_data(input);
    printf("weights: ");
    print_tensor_shape(self->weights);
    print_tensor_data(self->weights);
    printf("z: ");
    print_tensor_shape(z);
    print_tensor_data(z);
    printf("bias: ");
    print_tensor_shape(self->bias);
    print_tensor_data(self->bias);
    printf("output: ");
    print_tensor_shape(output);
    print_tensor_data(output);
    printf("\n");

    tensor_free(z);
    
    self->output = output;
    return output;
}

void dense_backward(Layer *self, Tensor *grad_output)
{
    // Calculate grad_weights using tensor_matmul and then call its backward pass
    Tensor *transposed_input = tensor_transpose(self->input, (int[]){1, 0});
    self->grad_weights = tensor_matmul(transposed_input, grad_output);
    tensor_matmul_backward(self->grad_weights, grad_output->grad);  // This propagates the gradient to the input and weights
    tensor_free(transposed_input);

    // Calculate grad_bias using tensor_sum and then call its backward pass
    self->grad_bias = tensor_sum(grad_output, 0);
    tensor_sum_backward(self->grad_bias, grad_output->grad);  // This propagates the gradient to the input

    // Calculate grad_input using tensor_matmul and then call its backward pass
    Tensor *transposed_weights = tensor_transpose(self->weights, (int[]){1, 0});
    self->grad_input = tensor_matmul(grad_output, transposed_weights);
    tensor_matmul_backward(self->grad_input, grad_output->grad);  // This propagates the gradient to the input and weights
    tensor_free(transposed_weights);

    // Optionally, free the gradients of grad_output if they're no longer needed
    tensor_free(grad_output);
}

void dense_update_params(Layer *self, Optimizer *optimizer)
{
    optimizer->update(optimizer, self->weights, self->grad_weights, self->bias, self->grad_bias);
}

void dense_free(Layer *self)
{
    free(self);
}