#include "layer.h"
#include "activation_functions.h"
#include "matrix_operations.h"
#include <vector>
#include <string>
#include <iostream>


Layer::Layer(int _num_neurons, std::string layer_id){
    this->num_neurons = _num_neurons;
    this->neurons.resize(this->num_neurons);

    this->layer_identity = layer_id;
    for(int i = 0; i < this->num_neurons; i++){
        this->neurons[i].input = 1;
    }
}

void Layer::initialize_weights(int r, int c){
    this->weights = random_vector(r, c);
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

void Layer::set_neuron_inputs(std::vector<double> in){
    for(int i = 0; i < this->neurons.size(); i++){
        this->neurons[i].input = in[i];
    }
}

std::vector<double> Layer::get_neuron_inputs(){
    std::vector<double> neuron_inputs;
    neuron_inputs.resize(this->num_neurons);
    for(int i = 0; i < this->neurons.size(); i++){
        neuron_inputs[i] = this->neurons[i].input; 
    }
    return neuron_inputs;
}

void Layer::set_neuron_outputs(){
    for(int i = 0; i < this->neurons.size(); i++){
        this->neurons[i].output = forward_activation_function(this->neurons[i].input);
   }
}

std::vector<double> Layer::get_neuron_outputs(){
    std::vector<double> neuron_outputs; 
    neuron_outputs.resize(this->num_neurons);
    for(int i = 0; i < this->num_neurons; i++){
        neuron_outputs[i] = this->neurons[i].output; 
    }
    return neuron_outputs;
}

std::vector< std::vector<double> > Layer::forward(){

    std::vector< std::vector<double> > layer_in; layer_in.resize(1);
    std::vector< std::vector<double> > layer_out; //layer_out.resize(1); layer_out[0].resize(this->neurons.size());
    std::vector< std::vector<double> > layer_bias; 
    //std::cout << "\n.1.\n";
    /*if(this->layer_identity == "input"){
        for(auto n : this->neurons){
            layer_in[0].push_back(n.input);
        }
    }else{
        for(auto n : this->neurons){
            layer_in[0].push_back(n.output);
        }
    }*/

    //std::cout << "\n.2.\n";
    /*
    ADD BIAS
    for(auto n : this->neurons){
        layer_bias[0].push_back(n.bias);
    }*/
    //std::cout << "\n.3.\n";
    
    for(auto n : this->neurons){
        if(this->layer_identity == "input"){
            layer_in[0].push_back(n.input);    
        }else{
            //layer_in[0].push_back(forward_activation_function(n.input));
            layer_in[0].push_back(n.output);
        }
    }

    layer_out = multiply(layer_in, this->weights);
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

    //vector_add(layer_out, layer_bias);
    //std::cout << "\n.4.\n";
    
    /*
    for(int i = 0; i < layer_out[0].size(); i++){
        layer_out[0][i] = forward_activation_function(layer_out[0][i]);
    }*/

    //std::cout << "\n.5.\n";
    return layer_out;
}


void Layer::backward(){
    int i = 0;
}