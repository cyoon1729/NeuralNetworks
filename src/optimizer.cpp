#pragma once
#include <functional>
#include <string> 
#include "../include/network.h"
#include "../include/optimizer.h"
#include "../include/core_math.h"
#include "../include/error_function.h"

namespace optimizer{
// gradient descent

void VGD::define_error_function(){
    if(this->error_function_name == "MSE"){
        this->error_function = MSE;
        this->error_function_derivative = MSE_derivative;
    }else if(this->error_function_name == "cross_entropy"){
        this->error_function = cross_entropy;
        this->error_function_derivative = cross_entropy_derivative;
    }
}

void VGD::fit(neuralnet::MLP net, const std::vector<tensor::Tensor> &input, const std::vector<tensor::Tensor> &actual, const size_t batch_size){
    std::vector<tensor::Tensor> accumulated_gradients;
    for(size_t data = 0; data < batch_size; ++data){
        tensor::Tensor output = net.forward(input[data]);
        std::vector<tensor::Tensor> activated_layers = net->get_activated_layers();
        
        // transpose activated values
        for(auto &tensors : activated_layers){
            tensors.T();
        }
        
        for(size_t i = net->num_layer-2; i >= 0; --i){
            if(accumulated_gradients.size() != net->num_layer){
                accumulated_gradients.append(delta[i + 1] * activated_layers[i]);
            }else{
                accumulated_gradients[i] = accumulated_gradients[i] + delta[i+1] * activated_layers[i];
            }
        }

        for(auto &tensors : accumulated_gradients){
            tensors = (1 / batch_size)(double) * tensors;   
        }






    }

}
}

