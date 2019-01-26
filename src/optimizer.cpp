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
    }/*else if(this->error_function_name == "cross_entropy"){
        this->error_function = cross_entropy;
        this->error_function_derivative = cross_entropy_derivative;
    }*/
}

void VGD::step(neuralnet::MLP net, const std::vector< std::vector<double> > &dataset, const std::vector< std::vector<double> > &actual, const size_t batch_size){
    this->define_error_function();
    std::vector<tensor::Tensor> accumulated_gradients;
    std::vector<tensor::Tensor> partial_wrt_weight;
    tensor::Tensor output;
    for(size_t input = 0; input < batch_size; ++input){
        output = net.forward(dataset[input]);
        std::vector<tensor::Tensor> activated_layers = net->get_activated_layers();
        
        // transpose activated values
        for(auto &tensors : activated_layers){
            tensors.T();
        }
        
        for(size_t i = net->num_layer-2; i >= 0; --i){
            if(accumulated_gradients.size() != net->num_layer){
                accumulated_gradients.push_back(delta[i + 1] * activated_layers[i]);
            }else{
                accumulated_gradients[i] = accumulated_gradients[i] + delta[i+1] * activated_layers[i];
            }
        }
    }
    partial_wrt_weight = accumulated_gradients;

    // batch coefficient and momentum 
    for(size_t index = 0; index < net->num_layer; ++index){
        partial_wrt_weight[i] = (1 / batch_size) * partial_wrt_weight[i] + this->momentum * net->get_layer_weights(net->num_layer - i - 1);
    }

    // multiply learning rate
    for(auto &tensors : partial_wrt_weight){
        tensors = this->learning_rate * tensors;
    }

    net.update_weights(partial_wrt_weight);
}

}

