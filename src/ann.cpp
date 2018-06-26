#include "../include/ann.h"
#include "../include/layer.h"
#include "../include/activation_functions.h"
#include "../include/matrix_operations.h"
#include <string.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

ANN::ANN(int _num_inputs, int _num_hidden_layers, int _num_hidden_neurons, int _num_outputs){
    
    //seed random for weight initialization
    srand(time(NULL));    
    
    //neural network parameters
    this->num_inputs = _num_inputs;
    this->num_hidden_layers = _num_hidden_layers;
    this->num_hidden_neurons = _num_hidden_neurons;
    this->num_outputs = _num_outputs;

    //initialize input layer
    Layer input_layer(this->num_inputs, "input");
    input_layer.initialize_weights(this->num_inputs, this->num_hidden_neurons);
    this->layers.push_back(input_layer);

    //initialize hidden layers
    for(int i = 0; i < this->num_hidden_layers; i++){
        Layer hidden_layer(this->num_hidden_neurons, "hidden");
        this->layers.push_back(hidden_layer);
    }

    //initialize output layer
    Layer output_layer(this->num_outputs, "output");
    this->layers.push_back(output_layer);

    //initialize weight matrices
    for(int i = 0; i <= this->num_hidden_layers; i++){
        this->layers[i].initialize_weights(layers[i + 1].return_num_neurons(), layers[i].return_num_neurons());
        this->layers[i].initialize_delta_weights(layers[i + 1].return_num_neurons(), layers[i].return_num_neurons());
    }

    std::cout << "Neural network initialized\n";
}


void ANN::set_learning_rate(double x){
    for(int i = 0; i < this->layers.size(); i++){
        layers[i].set_learningRate(x);
    }
    this->learning_rate = x;
}

void ANN::set_momentum(double x){
    for(int i = 0; i < this->layers.size(); i++){
        layers[i].set_Lmomentum(x);
    }
}

void ANN::set_activation_functions(std::vector<std::string> _activation_functions){
    this->activationFunctions = _activation_functions;
    for(int i = 1; i <= this->num_hidden_layers + 1; i++){
        this->layers[i].set_activation_function(this->activationFunctions[i-1]);
    }
}

std::vector<double> ANN::run(std::vector<double> input){
    std::vector<double> nn_output; 
    std::vector< std::vector<double> > layer_output; 
  
    //input layer
    this->layers[0].z_values(input);
    layer_output = this->layers[0].forward_propagate(); 
    
    //hidden layers and output layer
    for(int i = 1; i <= num_hidden_layers; i++){
    
        this->layers[i].z_values(layer_output[0]);
        this->layers[i].activation_values();
        
        layer_output = this->layers[i].forward_propagate();
    }  

    //run output layer in activation function
    this->layers[this->num_hidden_layers+1].z_values(layer_output[0]);
    this->layers[this->num_hidden_layers+1].activation_values();
    nn_output = this->layers[this->num_hidden_layers+1].get_activation_values();
    
    std::cout << "forward pass complete\n";

    this->outputs = nn_output;
    return nn_output;
}

double ANN::cross_entropy_error(std::vector< std::vector<double> > target, std::vector< std::vector<double> > output, int batch_size){
    double cross_entropy = 0;

    for(int i = 0; i < this->num_outputs; i++){
        cross_entropy += target[0][i] * std::log(output[0][i]) + (1 - target[0][i] * std::log((1-output[0][i])));
    }
    cross_entropy *= -(1 / batch_size);

    return cross_entropy;
}

double ANN::mean_square_error(std::vector< std::vector<double> > target, std::vector< std::vector<double> > output, int batch_size){
    double mse = 0;
    for(int i = 0; i < this->num_outputs; i++){
        mse += std::pow(target[0][i] - output[0][i], 2);
    }
    mse = mse / 2;
    return mse;
}

void ANN::train(std::vector<double> input, std::vector<double> expected_output){
    this->inputs = input;
    std::vector< std::vector<double> > nn_input; nn_input.resize(1); nn_input[0] = this->inputs;
    std::vector< std::vector<double> > nn_output; nn_output.resize(1); 
    std::vector< std::vector<double> > _expected_output; _expected_output.resize(1);
    _expected_output[0] = expected_output;
    
    for(int epoch = 0; epoch < this->max_epochs; epoch++){
        nn_output[0] = run(nn_input[0]);

        //batch size = 1 for online learning
        this->error = mean_square_error(_expected_output, nn_output, 1);     
        if(this->error <= this->desired_error){
            break;
        }
        
        std::vector< std::vector<double> > delta_outputLayer;
        std::vector< std::vector<double> > output_layer_Dvals;
        std::vector< std::vector<double> > output_layer_inputs;
        
        output_layer_Dvals.resize(1); output_layer_Dvals[0].resize(this->num_outputs);
        output_layer_inputs.resize(1); output_layer_inputs[0] = this->layers[this->num_hidden_layers+1].get_z_values();
        for(int i = 0; i < this->num_outputs; i++){
            double z = output_layer_inputs[0][i];
            output_layer_Dvals[0][i] = this->layers[this->num_hidden_layers+1].backward_activation_function(z);
        }
        
        delta_outputLayer = element_wise_multiply(element_wise_subtract(nn_output, _expected_output), output_layer_Dvals);
        this->layers[this->num_hidden_layers+1].set_delta(delta_outputLayer);
        
        for(int i = this->num_hidden_layers; i > 0; i--){
            this->layers[i].back_propagate(this->layers[i+1].get_delta());
            this->layers[i].update_weights();    
        }

        //input layer -> hidden layer
        std::vector< std::vector<double> > delta_inputLayer;
        delta_inputLayer = multiply(this->layers[1].get_delta(), nn_input);
        this->layers[0].set_delta(delta_inputLayer);
        this->layers[0].set_delta_weights(element_wise_add(scalar_multiply(this->learning_rate, delta_inputLayer), scalar_multiply(this->momentum, this->layers[0].get_delta_weights())));
        this->layers[0].update_weights(); 
    }
}

//read README for algorithm (but not written yet)
void ANN::batch_learn(std::vector< std::vector<double> > inputs, std::vector< std::vector<double> > outputs){
    
}

