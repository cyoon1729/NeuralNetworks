#include "../include/error_function.h"

namespace optimizer{

// mean squared error
tensor::Tensor MSE(tensor::Tensor &prediction, tensor::Tensor &actual){
    assert(prediction.m == 1 && actual.m == 1);
    tensor::Tensor error = actual - prediction;
    //square error
    error = 1/2 * error * error;
    return error;
}

// derivative of mean squared error with respect to output
tensor::Tensor MSE_derivative(tensor::Tensor &prediction, tensor::Tensor &actual){
    return actual - prediction;
}

}