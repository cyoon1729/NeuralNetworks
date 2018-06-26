#include "../include/layer.h"
#include "../include/activation_functions.h"
#include "../include/matrix_operations.h"
#include <vector>
#include <string>
#include <iostream>

Layer::Layer(int _num_neurons, std::string layer_id){
    this->num_neurons = _num_neurons;
    this->neurons.resize(this->num_neurons);
    this->layer_identity = layer_id;
}

void Layer::initialize_weights(int r, int c){
    this->weights = random_vector(r, c);
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

    std::cout << "========layer: " << this->layer_identity << "=======\n";
    std::cout << "------input------\n";
    for(int i = 0; i < layer_in.size(); i++){
        for(int j = 0; j < layer_in[0].size(); j++){
            std::cout << layer_in[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "-------weights------\n";
    for(int i = 0; i < this->weights.size(); i++){
        for(int j = 0; j < this->weights[0].size(); j++){
            std::cout << weights[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "------output------\n";
    for(int i = 0; i < layer_out.size(); i++){
        for(int j = 0; j < layer_out[0].size(); j++){
            std::cout << layer_out[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "===================================================\n";

    return transpose(layer_out);
}


void Layer::back_propagate(std::vector< std::vector<double> > p_delta){
    std::vector< std::vector<double> > _delta;
    std::vector< std::vector<double> > d_z_vals; 
    std::vector< std::vector<double> > activation_vals; 

    d_z_vals.resize(1); d_z_vals[0].resize(this->num_neurons);
    for(int i = 0; i < this->num_neurons; i++){
        double _z = this->neurons[i].z;
        d_z_vals[0][i] = backward_activation_function(_z);
    }

    activation_vals.resize(1); activation_vals[0].resize(this->num_neurons);
    for(int i = 0; i < this->num_neurons; i++){
        activation_vals[0][i] = neurons[i].z;    
    }
    
    d_z_vals = transpose(d_z_vals);
    _delta = element_wise_multiply(multiply(transpose(this->weights), p_delta), d_z_vals); 
    this->delta = _delta;
    this->delta_weights = element_wise_add(scalar_multiply(this->learning_rate, multiply(p_delta, activation_vals)), scalar_multiply(this->momentum, this->delta_weights));
}

void Layer::update_weights(){
    this->weights = element_wise_subtract(this->weights, this->delta_weights);
}