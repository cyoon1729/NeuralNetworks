#pragma once

#include "layer.h"
#include <vector>
#include <string>
#include <iostream>


class ANN{
        public:
            ANN(int _num_inputs, int _num_hidden_layers, int _num_hidden_neurons, int _num_outputs);

            void set_activation_functions(std::vector<std::string> _activation_functions);

            std::vector<double> run(std::vector<double> input);
            
            void train(std::vector<double> input, std::vector<double> expected_output);

        private:
            int num_inputs;
            int num_hidden_layers;
            int num_hidden_neurons;
            int num_outputs;

            //activation functions for hidden layers and output layer
            std::vector<std::string> activationFunctions;

            std::vector<Layer> layers;
};
