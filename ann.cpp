#include "ann.h"
#include "layer.h"
#include "activation_functions.h"
#include "matrix_operations.h"
#include <string.h>
#include <vector>

ANN::ANN(int _num_inputs, int _num_hidden_layers, int _num_hidden_neurons, int _num_outputs){
    std::cout << "Neural network initializing\n";
    this->num_inputs = _num_inputs;
    this->num_hidden_layers = _num_hidden_layers;
    this->num_hidden_neurons = _num_hidden_neurons;
    this->num_outputs = _num_outputs;

    Layer input_layer(this->num_inputs, "input");
    input_layer.initialize_weights(this->num_inputs, this->num_hidden_neurons);
    this->layers.push_back(input_layer);


    for(int i = 0; i < this->num_hidden_layers; i++){
        Layer hidden_layer(this->num_hidden_neurons, "hidden");
        if(i == this->num_hidden_layers - 1){
            hidden_layer.initialize_weights(this->num_hidden_neurons, this->num_outputs);
        }else{
            hidden_layer.initialize_weights(this->num_hidden_neurons, this->num_hidden_neurons);
        }
        this->layers.push_back(hidden_layer);
    }


    Layer output_layer(this->num_outputs, "output");
    this->layers.push_back(output_layer);

    std::cout << "Neural network initialized\n";
}

void ANN::set_activation_functions(std::vector<std::string> _activation_functions){
    //assert(_activation_functions.size() == _num_hidden_layers + 1);
    this->activationFunctions = _activation_functions;
    for(int i = 1; i <= this->num_hidden_layers + 1; i++){
        this->layers[i].set_activation_function(this->activationFunctions[i-1]);
    }
}

std::vector<double> ANN::run(std::vector<double> input){
    std::vector<double> nn_output; //output of neural network
    std::vector< std::vector<double> > la_output; //output of each layer
    std::vector<Neuron> layer_neurons;
    
    //input layer
    this->layers[0].set_neuron_inputs(input);
    la_output = this->layers[0].forward(); 
    //std::cout << "\ninput layer forward\n";

    //hidden layers and output layer
    for(int i = 1; i <= num_hidden_layers; i++){
        
        //std::cout << "hidden layer forward\n";

        this->layers[i].set_neuron_inputs(la_output[0]);
        this->layers[i].set_neuron_outputs();
        //this->layers[i].set_neuron_outputs();
        
        la_output = this->layers[i].forward();
        /*
        if(this->layers[i].return_layer_identity() != "output"){
            la_output = this->layers[i].forward();
        }
        */
    }  

    //run output layer in activation function
    this->layers[this->num_hidden_layers+1].set_neuron_inputs(la_output[0]);
    this->layers[this->num_hidden_layers+1].set_neuron_outputs();
    nn_output = this->layers[this->num_hidden_layers+1].get_neuron_outputs();
    
    //this->layers[this->num_hidden_layers+1].set_neuron_outputs();
    //nn_output = this->layers[this->num_hidden_layers+1].get_neuron_outputs();
    
    /*for(auto ne : this->layers[this->num_hidden_neurons + 1].neurons){
        double opt = this->layers[this->num_hidden_neurons + 1].forward_activation_function(ne.input);
        output.push_back(opt);
    }*/
    
    std::cout << "forward pass complete\n";
    return nn_output;
}

void train(){
    int i = 0;   
}

