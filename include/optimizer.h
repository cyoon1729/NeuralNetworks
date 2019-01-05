#pragma once
#include "network.h"
#include "core_math.h"
#include <functional>
#include <string> 

namespace optimizer{
// gradient descent
class VGD{
public:
    VGD(nueralnet::BaseNetwork &net, const std::string loss): nework(net) {
        define_error_function(loss);
    };
    void fit(neuralnet::MLP net, const std::vector<tensor::Tensor> &input, const std::vector<tensor::Tensor> &actual, const size_t batch_size);
    void define_error_function();

private:
    neuralnet::BaseNetwork network;
    const std::string error_function_name;
    const std::function<tensor::Tensor(tensor::Tensor &prediction, tensor::Tensor &actual)> error_function;
    const std::function<tensor::Tensor(tensor::Tensor &prediction, tensor::Tensor &actual)> error_function_derivatice;

};

class VGDmomentum{

};

class Adam{

};

class SGD{

};

}