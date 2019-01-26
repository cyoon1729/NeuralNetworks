#include <cmath>
#include <assert.h>
#include "./core_math.h"

namespace optimizer{

// mean squared error
tensor::Tensor MSE(tensor::Tensor &prediction, tensor::Tensor &actual);

// derivative of mean squared error with respect to output
tensor::Tensor MSE_derivative(tensor::Tensor &prediction, tensor::Tensor &actual);

// cross entropy error
//tensor::Tensor cross_entropy(tensor::Tensor &prediction, tensor::Tensor &actual);

// derivative of cross entropy error with respect to output
//tensor::Tensor cross_entropy_derivative(tensor::Tensor &prediction, tensor::Tensor &actual);

};