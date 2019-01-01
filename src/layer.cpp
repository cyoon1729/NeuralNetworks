#include "../include/layer.h"
#include "../include/matrix_operations.h"
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
    for(auto& z : this->activated_neurons.matrix){
        z = this->activation_function(z);
    }
}
void Layer::forward(Layer &next_layer){
    matrix::copy_matrix(this->transposed_weights, this->weights);
    transposed_weights.T();
    this->activate();
    next_layer.neurons = this->transposed_weights * this->activated_neurons + this->bias;
    std::cout << this->activated_neurons;
    std::cout << "\n";
    std::cout << next_layer.neurons;
}

// compute gradient of activation fuction 
matrix::Matrix Layer::gradients(){
    matrix::Matrix gradients = this->neurons;
    for(auto& z : gradients.matrix){
        z = this->activation_function_derivative(z);
    }
    return gradients;
}

matrix::Matrix Layer::get_weights(){
    return this->weights;
}

void Layer::feed(std::vector<double> &input){
    assert(input.size() == this->fan_in);
    this->neurons.matrix = input;
    std::cout << this->neurons;
}
/*void pass_forward(Layer& lhs, Layer& rhs){
    lhs->activate();
    rhs.neurons = lhs.transposed_weights * lhs.activated_neurons + lhs.bias
}*/
}



/*
Layer::Layer(int _num_neurons, std::string layer_id){
    this->num_neurons = _num_neurons;
    this->neurons.resize(this->num_neurons);
    this->layer_identity = layer_id;
}

void Layer::initialize_weights(int r, int c){
    this->weights = random_vector(r, c, *this);
}

void Layer::initialize_delta_weights(int r, int c){
    this->delta_weights = zero_vector(r, c);
}

void Layer::set_activation_function(std::string _activation_function){
    this->activation_funct = _activation_function;
}

double Layer::forward_activation_function(double x){
    if(this->activation_funct == "sigmoid") return sigmoid(x);
    else if(this->activation_funct == "tanh") return tanh(x);
    else if(this->activation_funct == "relu") return relu(x);
}

double Layer::backward_activation_function(double x){
    if(this->activation_funct == "sigmoid") return d_sigmoid(x);
    else if(this->activation_funct == "tanh") return d_tanh(x);
    else if(this->activation_funct == "relu") return d_relu(x);
}

void Layer::z_values(std::vector<double> in){
    for(int i = 0; i < this->neurons.size(); i++){
        this->neurons[i].z = in[i];
    }
}

std::vector<double> Layer::get_z_values(){
    std::vector<double> z_vals;
    z_vals.resize(this->num_neurons);
    for(int i = 0; i < this->neurons.size(); i++){
        z_vals[i] = this->neurons[i].z; 
    }
    return z_vals;
}

void Layer::activation_values(){
    for(int i = 0; i < this->neurons.size(); i++){
        this->neurons[i].a = forward_activation_function(this->neurons[i].z);
   }
}

std::vector<double> Layer::get_activation_values(){
    std::vector<double> a_vals; 
    a_vals.resize(this->num_neurons);
    for(int i = 0; i < this->num_neurons; i++){
        a_vals[i] = this->neurons[i].a; 
    }
    return a_vals;
}

std::vector< std::vector<double> > Layer::forward_propagate(){

    std::vector< std::vector<double> > layer_in; layer_in.resize(1);
    std::vector< std::vector<double> > layer_out; 
    std::vector< std::vector<double> > layer_bias; 
    
    for(auto n : this->neurons){
        if(this->layer_identity == "input"){
            layer_in[0].push_back(n.z);    
        }else{
            layer_in[0].push_back(n.a);
        }
    }
    layer_out = multiply(this->weights, transpose(layer_in));
    
    //add bias
    return transpose(layer_out);
}

void Layer::update_weights(){
    this->weights = element_wise_subtract(this->weights, scalar_multiply(this->learning_rate, this->weights_update));
}
*/