#include "../include/layer.h"
#include "../include/core_math.h"
#include "../include/activation_functions.h"
#include <vector>
#include <string>
#include <iostream>
#include <assert.h>

namespace neuralnet{

void Layer::set_activation_function(){
    if(this->activation_function_name == "linear"){
        this->activation_function = linear;
        this->activation_function_derivative = linear_derivative;
    }else if(this->activation_function_name == "sigmoid"){
        this->activation_function = sigmoid;
        this->activation_function_derivative = sigmoid_derivative;
    }else if(this->activation_function_name == "relu"){
        this->activation_function = relu;
        this->activation_function_derivative = relu;
    }else if(this->activation_function_name == "tanh"){
        this->activation_function = hyperbolic_tan;
        this->activation_function_derivative = hyperbolic_tan_derivative;
    }else{
        std::cout << "activation function is not recognized, setting to linear";
        this->activation_function = linear;
        this->activation_function_derivative = linear_derivative;
    }
}

// perform forward propagate from layer to next layer
void Layer::activate(){
    this->activated_neurons = this->neurons;
    for(auto& z : this->activated_neurons.tensor){
        z = this->activation_function(z);
    }
}
void Layer::pass_forward(Layer &next_layer){
    //tensor::copy_tensor(this->transposed_weights, this->weights);
    // transposed_weights.T();
    if(!this->weights.transposed){
        this->weights.T();
    }
    this->activate();
    next_layer.neurons = this->weights * this->activated_neurons + this->bias;
    // std::cout << this->activated_neurons;
    // std::cout << "\n";
    // std::cout << next_layer.neurons;
}

// compute gradient of activation fuction 
tensor::Tensor Layer::gradients(){
    tensor::Tensor gradients = this->neurons;
    for(auto& z : gradients.tensor){
        z = this->activation_function_derivative(z);
    }
    return gradients;
}

tensor::Tensor Layer::get_weights(){
    return this->weights;
}

void Layer::step(tensor::Tensor &weights_update){
    this->weights = this->weights - weights_update;
}

void Layer::feed(std::vector<double> &input){
    assert(input.size() == this->fan_in);
    this->neurons.tensor = input;
    // std::cout << this->neurons;
}

tensor::Tensor Layer::get_activated_neurons(){
    return this->activated_neurons;
}
}
