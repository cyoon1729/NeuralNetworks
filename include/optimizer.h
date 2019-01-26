#pragma once
#include "network.h"
#include "core_math.h"
#include "error_function.h"

#include <functional>
#include <string> 
#include <iostream>

namespace optimizer{
// gradient descent
class VGD{
public:
    VGD(neuralnet::BaseNetwork &net, const std::string loss): network(net), error_function_name(loss) {};

    void step(neuralnet::MLP net, const std::vector< std::vector<double> > &dataset, const std::vector< std::vector<double> > &actual, const size_t batch_size);
    void define_error_function();

private:
    neuralnet::BaseNetwork network;
    const std::string error_function_name;
    const std::function< tensor::Tensor tensor::Tensor &prediction, tensor::Tensor &actual) > error_function;
    const std::function< tensor::Tensor(tensor::Tensor &prediction, tensor::Tensor &actual) > error_function_derivative;
    
};

class VGDmomentum{

};

class Adam{

};

class SGD{

};

}