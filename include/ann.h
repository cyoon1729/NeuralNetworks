#pragma once

#include "layer.h"
#include <vector>
#include <string>
#include <iostream>


class ANN{
    public:
        ANN(int _num_inputs, int _num_hidden_layers, int _num_hidden_neurons, int _num_outputs);

        //set learning rate for all layers
        void set_learning_rate(double x);
        
        //set momentum for all layers
        void set_momentum(double x);   
        
        inline void set_desired_error(double x){
            this->desired_error = x;
        }

        inline void set_max_epochs(int x){
            this->max_epochs = x;
        }

        double cross_entropy_error(std::vector< std::vector<double> > target, std::vector< std::vector<double> > output, int batch_size);

        double mean_square_error(std::vector< std::vector<double> > target, std::vector< std::vector<double> > output, int batch_size);

        void set_activation_functions(std::vector<std::string> _activation_functions);

        std::vector<double> run(std::vector<double> input);
        
        void train(std::vector<double> input, std::vector<double> expected_output);

        void batch_learn(std::vector< std::vector<double> > inputs, std::vector< std::vector<double> > outputs);

        inline double ret_error(){
            return this->error;
        }

    private:
        int num_inputs;
        int num_hidden_layers;
        int num_hidden_neurons;
        int num_outputs;
        int max_epochs;
        double learning_rate;
        double momentum;
        double desired_error;
        double error;
        
        std::vector<std::string> activationFunctions;
        std::vector<Layer> layers;
        std::vector<double> inputs;
        std::vector<double> outputs;
        
};
