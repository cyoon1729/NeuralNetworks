#pragma once
#include <vector>
#include <cmath>
#include <string>
#include "../include/layer.h"
#include "../include/ann.h"
#include "../include/matrix_operations.h"

ANN backprop(ANN neuralnet, std::vector<double> target){
    std::vector< std::vector<double> > inputs; inputs.resize(1);
    std::vector< std::vector<double> > prediction; prediction.resize(1);
    std::vector< std::vector<double> > expected_output; expected_output.resize(1);


    inputs[0] = neuralnet.return_inputs();

    prediction[0] = neuralnet.return_outputs();

    expected_output[0] = target;


    //output layer
    std::vector< std::vector<double> > delta_output;
    std::vector< std::vector<double> > output_z; output_z.resize(1);
    std::vector< std::vector<double> > output_dz; output_dz.resize(1); output_dz[0].resize(neuralnet.num_outputs);
    output_z[0] = neuralnet.layers[neuralnet.num_hidden_layers+1].get_z_values();
    for(int i = 0; i < neuralnet.num_outputs; i++){
        double dz = neuralnet.layers[neuralnet.num_hidden_layers+1].backward_activation_function(output_z[0][i]);
        output_dz[0][i] = dz;
    }
    delta_output = element_wise_multiply(element_wise_subtract(prediction, expected_output), output_dz);

    neuralnet.layers[neuralnet.num_hidden_layers + 1].set_delta(delta_output);

    
    //hidden layers
    std::vector< std::vector<double> > delta_hidden;
    for(int i = neuralnet.num_hidden_layers; i >= 1; i--){
        std::vector< std::vector<double> > d_z; d_z.resize(1);
        for(auto x : neuralnet.layers[i].get_z_values()){
            double dz = neuralnet.layers[i].backward_activation_function(x);
            d_z[0].push_back(dz);
        }
        delta_hidden = element_wise_multiply(transpose(multiply(transpose(neuralnet.layers[i].weights), transpose(neuralnet.layers[i+1].delta))), d_z);

        neuralnet.layers[i].set_delta(delta_hidden);
    }

    //input layer;
    neuralnet.layers[0].delta_weights = element_wise_add(neuralnet.layers[0].delta_weights, multiply(transpose(neuralnet.layers[1].delta), inputs));
    
    for(int i = 1; i <= neuralnet.num_hidden_layers; i++){
        std::vector< std::vector<double> > layer_a_vals; layer_a_vals.resize(1);
        layer_a_vals[0] = neuralnet.layers[i].get_activation_values();
        neuralnet.layers[i].delta_weights = element_wise_add(neuralnet.layers[i].delta_weights, multiply(transpose(neuralnet.layers[i+1].delta), layer_a_vals));  
    }

    return neuralnet;
}